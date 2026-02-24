import torch
import torch.nn.functional as F
import math

def get_attention_and_lse(query, key, value, causal=False, sm_scale=None):
    """
    通用 Attention 计算，返回局部输出 O_i 和 LSE_i。
    (如果使用 GPU，这里可以替换为 flash_attn_func 且 return_lse=True)
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(query.shape[-1])
        
    scores = torch.matmul(query, key.transpose(-2, -1)) * sm_scale
    if causal:
        q_len, k_len = query.shape[-2], key.shape[-2]
        mask = torch.full((q_len, k_len), float('-inf'), device=query.device)
        mask = torch.triu(mask, diagonal=k_len - q_len + 1)
        scores += mask.unsqueeze(0).unsqueeze(0)
    
    # 获取局部自由能配分函数 LSE_i
    lse = torch.logsumexp(scores, dim=-1)
    
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, value)
    return out, lse

def memory_efficient_stage_3_and_4(out_prefix, lse_prefix, q_query, k_cache_chunks, v_cache_chunks, out_query, lse_query, k_top=2):
    """
    为了解决 96GB OOM 专门重构的分块验证 (Chunk-by-Chunk Reduction)。
    不再强行拼接巨型 KV Cache，而是对每个 Chunk 计算局部输出 O_i 和 LSE_i，
    并在内存中直接完成非对称惩罚与全局归约。
    (已回退：由于物理删除/缓存驱逐存在极大的信息永久丢失和 Attention Sinks 崩溃风险，
    此处保留全部 Chunk 的 KV cache，纯粹通过数学上的 Safe Softmax 抑制噪音)
    
    参数说明:
    q_query: 用户问题的 Query 张量
    k_cache_chunks, v_cache_chunks: 包含 N 个子文档的 KV Tensor 列表
    out_prefix, lse_prefix: System Prompt / Prefix chunk.
    out_query, lse_query: User Query chunk (自身的 causal 注意力).
    k_top: 提取健康基线的 Top-K 参数。
    """
    num_chunks = len(k_cache_chunks)
    
    # 步骤 A：分块前向（获取全部 Chunk 的 O_i 和 LSE_i）
    # 逐块进行计算可以避免巨型 Tensor 的 OOM，但所有 chunk 均保留在内存中供未来 Token 使用。
    out_docs = []
    lse_docs = []
    for i in range(num_chunks):
        k_chunk = k_cache_chunks[i]
        v_chunk = v_cache_chunks[i]
        # Query 对每一个 Chunk 单独做 Cross-Attention
        out_i, lse_i = get_attention_and_lse(q_query, k_chunk, v_chunk, causal=False)
        out_docs.append(out_i)
        lse_docs.append(lse_i)
        
    # Stack 起来用于 Top-K 的数学运算 [num_chunks, bsz, heads, q_len]
    out_docs = torch.stack(out_docs, dim=0)
    lse_docs = torch.stack(lse_docs, dim=0)
    
    # ========================================================
    # 阶段 2：计算健康基线 (Target_LSE) 和赤字 (Delta)
    # ========================================================
    k_actual = min(k_top, num_chunks)
    topk_vals, _ = torch.topk(lse_docs, k_actual, dim=0)
    target_lse = topk_vals.mean(dim=0) # [bsz, heads, q_len]
    
    # \Delta_i = Target_LSE - LSE_doc_i
    delta_i = target_lse.unsqueeze(0) - lse_docs
    
    # ========================================================
    # 阶段 3：软性泄放与非对称平移 (Softplus Drain & Asymmetric Shift)
    # ========================================================
    # 计算 Offset: Offset_i = -F.softplus(\Delta_i)
    offset_i = -F.softplus(delta_i)
    
    # 核心：将 Offset 加到对应的 Context LSE 上
    lse_docs_tilde = lse_docs + offset_i
    # Prefix 和 Query 绝对不加 offset
    
    # ========================================================
    # 阶段 4：安全全局归约 (Safe Flash Decoding Reduction)
    # ========================================================
    lse_prefix_exp = lse_prefix.unsqueeze(0) # [1, bsz, heads, q_len]
    lse_query_exp  = lse_query.unsqueeze(0)  # [1, bsz, heads, q_len]
    
    # 拼接所有的 LSE_tilde [num_chunks + 2, bsz, heads, q_len]
    lse_concat = torch.cat([lse_prefix_exp, lse_docs_tilde, lse_query_exp], dim=0)
    
    # 计算全局 LSE_global
    lse_global = torch.logsumexp(lse_concat, dim=0) # [bsz, heads, q_len]
    
    # 计算混合权重 (Mixing Weights) W_i = exp(LSE_i - LSE_global)
    w_prefix = torch.exp(lse_prefix - lse_global).unsqueeze(-1)          
    w_docs = torch.exp(lse_docs_tilde - lse_global.unsqueeze(0)).unsqueeze(-1) 
    w_query = torch.exp(lse_query - lse_global).unsqueeze(-1)            
    
    # 输出组装: O_final = Σ(W_i * O_i)
    out_docs_sum = (w_docs * out_docs).sum(dim=0)
    out_final = (w_prefix * out_prefix) + out_docs_sum + (w_query * out_query)
    
import types
from functools import partial
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

class ResearchEngine:
    """
    负责将我们的新算法 (memory_efficient_stage_3_and_4)
    动态注入 (Monkey Patch) 到 Llama 的 Attention 层中。
    """
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        self.current_stage = None
        self.metadata = {}

    def patched_forward(self, layer_idx, module, hidden_states, position_ids, past_key_value=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        
        # 1. Standard Projections
        query_states = module.q_proj(hidden_states).view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

        cos, sin = module.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.current_stage == 'prefix' or self.current_stage == 'context':
            # 阶段 1/2: 正常的 Prefill (并行提特征)
            if self.current_stage == 'context' and past_key_value is not None:
                (pk, pv, pp) = past_key_value[0] # 这里简化，提取 batched_pre_cache 的结构
                key_states = torch.cat([pk, key_states], dim=2)
                value_states = torch.cat([pv, value_states], dim=2)
                position_ids = torch.cat([pp, position_ids], dim=-1)
            
            k_rpt = repeat_kv(key_states, module.num_key_value_groups)
            v_rpt = repeat_kv(value_states, module.num_key_value_groups)
            out, _ = get_attention_and_lse(query_states, k_rpt, v_rpt, causal=(q_len > 1))
            
            out = out.transpose(1, 2).reshape(bsz, q_len, module.hidden_size)
            out = module.o_proj(out)
            return out, None, (key_states, value_states, position_ids)

        elif self.current_stage == 'query':
            # 阶段 3: 并行探测与无损动态配分归约
            
            # 从 global_cache 拆解回 Prefix 和 N 个 Chunks (以避免拼接引起的 OOM)
            # 在 run_longbench.py 中装配的 global_cache 结构是 [(k_pre_ctx_concat, v, p, num_ctx, len_pre)...]
            gk, gv, gp, num_ctx, len_prefix = past_key_value
            
            k_prefix = repeat_kv(gk[:, :, :len_prefix], module.num_key_value_groups)
            v_prefix = repeat_kv(gv[:, :, :len_prefix], module.num_key_value_groups)
            
            out_prefix, lse_prefix = get_attention_and_lse(query_states, k_prefix, v_prefix, causal=False)
            
            # --- 核心改动：重新切分 Chunks，避免 OOM 暴死 ---
            k_ctx_flattened = gk[:, :, len_prefix:]
            v_ctx_flattened = gv[:, :, len_prefix:]
            chunk_length = k_ctx_flattened.shape[2] // num_ctx
            
            k_cache_chunks = []
            v_cache_chunks = []
            for i in range(num_ctx):
                # 切出独立 Chunk
                kc = k_ctx_flattened[:, :, i*chunk_length : (i+1)*chunk_length]
                vc = v_ctx_flattened[:, :, i*chunk_length : (i+1)*chunk_length]
                k_cache_chunks.append(repeat_kv(kc, module.num_key_value_groups))
                v_cache_chunks.append(repeat_kv(vc, module.num_key_value_groups))
                
            # Query 自己
            k_q_rpt = repeat_kv(key_states, module.num_key_value_groups)
            v_q_rpt = repeat_kv(value_states, module.num_key_value_groups)
            out_q, lse_q = get_attention_and_lse(query_states, k_q_rpt, v_q_rpt, causal=True)
            
            top_k = self.metadata.get('top_k', 2)
            
            # 执行纯数学无损软降噪：OOM-Safe 的分块处理
            attn_output = memory_efficient_stage_3_and_4(
                out_prefix, lse_prefix,
                query_states,
                k_cache_chunks, v_cache_chunks,
                out_q, lse_q,
                k_top=top_k
            )
            
            current_pos = gp.max().item() + 1
            query_pos = position_ids - position_ids.min().item() + current_pos
             
            new_cache = (torch.cat([gk, key_states], dim=2), 
                         torch.cat([gv, value_states], dim=2), 
                         torch.cat([gp, query_pos], dim=-1),
                         num_ctx, len_prefix)
            
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, module.hidden_size)
            attn_output = module.o_proj(attn_output)
            return attn_output, None, new_cache
        
        return self.original_forwards[layer_idx](hidden_states, position_ids=position_ids, past_key_value=past_key_value, **kwargs)

    def inject(self, stage, **kwargs):
        self.current_stage = stage
        self.metadata.update(kwargs)
        for i, layer in enumerate(self.model.model.layers):
            if i not in self.original_forwards:
                self.original_forwards[i] = layer.self_attn.forward
            layer.self_attn.forward = types.MethodType(partial(self.patched_forward, i), layer.self_attn)
