import torch
import torch.nn.functional as F
from engine import get_attention_and_lse, memory_efficient_stage_3_and_4

def run_test():
    torch.manual_seed(42)
    # 模拟张量维度
    bsz = 1
    heads = 4
    q_len = 16
    dim = 64
    num_chunks = 5
    
    print(f"[*] 模拟配置: Batch={bsz}, Heads={heads}, Query_Len={q_len}, Dim={dim}, Conetxt_Chunks={num_chunks}")
    
    q_query = torch.randn(bsz, heads, q_len, dim)
    
    # ====== 阶段 1 模拟：获取局部自由能 (Chunked Forward) ======
    print("[*] 阶段 1: Chunked Forward Pass - 提取 LSE 能量")
    # Prefix (System Prompt)
    k_pre, v_pre = torch.randn(bsz, heads, 100, dim), torch.randn(bsz, heads, 100, dim)
    out_prefix, lse_prefix = get_attention_and_lse(q_query, k_pre, v_pre, causal=False)
    print(f"  -> Prefix LSE   均值: {lse_prefix.mean().item():.4f}")
    
    # N 个 Context Chunks KV Caches
    k_cache_chunks = []
    v_cache_chunks = []
    for i in range(num_chunks):
        k, v = torch.randn(bsz, heads, 200, dim), torch.randn(bsz, heads, 200, dim)
        k_cache_chunks.append(k)
        v_cache_chunks.append(v)
    
    print(f"  -> Context 生成了 {num_chunks} 组离散的 KV Caches.")
    
    # User Query Chunk (Causal Attention toward itself)
    k_query, v_query = torch.randn(bsz, heads, q_len, dim), torch.randn(bsz, heads, q_len, dim)
    out_query, lse_query = get_attention_and_lse(q_query, k_query, v_query, causal=True)
    print(f"  -> Query (Causal) LSE    均值: {lse_query.mean().item():.4f}")
    
    print("\n" + "="*50)
    print("[*] 阶段 2 ~ 4: OOM-Safe Chunk-by-Chunk 自适应赤字与归约")
    # 核心算法验证 (已去掉风险极高的剪枝框架)
    out_final = memory_efficient_stage_3_and_4(
        out_prefix, lse_prefix,
        q_query,
        k_cache_chunks, v_cache_chunks,
        out_query, lse_query,
        k_top=2 # Top-K=2
    )
    
    print("="*50)
    print(f"[+] 计算完成！最终 Attention 输出维度: {out_final.shape} (预期维度: [{bsz}, {heads}, {q_len}, {dim}])")
    print(f"[+] 算法数值无 NaN/Inf? {not torch.isnan(out_final).any() and not torch.isinf(out_final).any()}")

if __name__ == "__main__":
    run_test()
