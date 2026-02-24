import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from engine import ResearchEngine

# --- 核心配置 ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# 如果没有 GPU，这也可以在 CPU 上跑通逻辑（就是慢）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_llama3_prompt(context, query):
    """
    根据 Llama-3 的官方格式构建输入。
    由于我们要切分 Context，系统提示和 Query 会分别作为 Prefix 和 Suffix。
    """
    system_prompt = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>\n"
    user_header = "<|start_header_id|>user<|end_header_id|>\n\n"
    
    # 按照 APE 的逻辑拆分：
    # Prefix: System prompt + User Header
    prefix = system_prompt + user_header
    # Contexts: The actual long document(s)
    contexts = context 
    # Query: The question + Assistant Header
    query = f"\n{query}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prefix, contexts, query

def run_longbench_demo():
    print("[*] 正在绕过网络限制，在本地构建模拟的长文本测试例...")
    
    # 模拟一个 Longbench/narrativeqa 的数据条目
    sample = {
        "input": "Where does the story take place?",
        "context": "The quick brown fox jumps over the lazy dog. " * 500 + "The story takes place in the mythical forest of Eldoria, known for its whispering trees and glowing streams. " + "The quick brown fox jumps over the lazy dog. " * 500,
        "answers": ["Eldoria"],
        "length": 10000
    }
    
    print(f"[*] 成功加载测试数据! 问题: {sample['input']}")
    print(f"[*] 文章原始长度: {sample['length']} words")
    
    prefix, document, query = build_llama3_prompt(sample['context'], sample['input'])
    
    print(f"[*] 初始化模型: {MODEL_ID} on {DEVICE}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 考虑到本地可能没那么大内存，以 float32 加载在 CPU 或者 Cuda
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map=DEVICE
        )
        engine = ResearchEngine(model)
    except Exception as e:
        print(f"[!] 模型加载失败 (你本地可能没有权重或者内存不够): {e}")
        print("[!] 但跑分脚本的逻辑已经在这里了，你随时可以带去服务器跑。")
        return

    # ==========================
    # 开始执行你的"能量赤字"算法推理
    # ==========================
    
    # 将超长的 document 切分成多个 Chunk (这里粗略按字符或者 Token 长度切分)
    # 假设我们每 2000 个字符做一个 chunk
    chunk_size = 2000
    contexts_chunks = [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]
    if len(contexts_chunks) > 5:
        contexts_chunks = contexts_chunks[:5] # 本地截断一下防止爆内存
        
    print(f"[*] 将超长文章切分为了 {len(contexts_chunks)} 个并行 Context Chunks.")

    # STAGE 1: Prefix
    print("[*] 阶段 1: 处理 Prefix...")
    engine.inject('prefix')
    p_ids = tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(DEVICE)
    p_pos = torch.arange(p_ids.shape[1]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(p_ids, position_ids=p_pos, use_cache=True)
    prefix_cache = out.past_key_values

    # STAGE 2: Parallel Contexts
    print("[*] 阶段 2: 并行处理所有 Context Chunks...")
    engine.inject('context')
    num_ctx = len(contexts_chunks)
    batched_pre_cache = []
    for pk, pv, pp in prefix_cache:
        batched_pre_cache.append((pk.repeat(num_ctx, 1, 1, 1), pv.repeat(num_ctx, 1, 1, 1), pp))
        
    c_ids = tokenizer(contexts_chunks, return_tensors='pt', padding=True, add_special_tokens=False).input_ids.to(DEVICE)
    c_pos = torch.arange(p_ids.shape[1], p_ids.shape[1] + c_ids.shape[1]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(c_ids, position_ids=c_pos, past_key_values=batched_pre_cache, use_cache=True)
    
    # 组装我们要交给 Query 的 Global Cache，准备进行 Stage 3 能量基准测试
    global_cache = []
    mask = (c_ids != tokenizer.pad_token_id).flatten()
    for layer_out in out.past_key_values:
        k, v, p = layer_out
        len_pre = p_ids.shape[1]
        k_pre, v_pre, p_pre = k[:1, :, :len_pre], v[:1, :, :len_pre], p[:, :len_pre]
        
        # 将 Contexts 组装回一整条线，你的算法引擎内部会提取出 Chunk 级别信息
        k_ctx = k[:, :, len_pre:].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
        v_ctx = v[:, :, len_pre:].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
        p_ctx = p[:, len_pre:].repeat(num_ctx, 1).flatten()[mask].unsqueeze(0)
        
        global_cache.append((torch.cat([k_pre, k_ctx], dim=2), 
                             torch.cat([v_pre, v_ctx], dim=2), 
                             torch.cat([p_pre, p_ctx], dim=1),
                             num_ctx, len_pre))

    # STAGE 3: Query & 生成
    print("[*] 阶段 3 & 4: 注入 Query，执行计算基线 -> 软性泄放 -> 安全归约...")
    engine.inject('query', top_k=2) # 你的参数：Target_LSE 取 Top 2
    
    q_ids = tokenizer(query, return_tensors='pt', add_special_tokens=False).input_ids.to(DEVICE)
    ctx_flat = c_ids.flatten()[mask].unsqueeze(0)
    full_ids = torch.cat([p_ids, ctx_flat, q_ids], dim=-1).to(torch.long)
    
    print("[*] 生成答案中...")
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=full_ids,
            past_key_values=global_cache,
            max_new_tokens=50,
            do_sample=False
        )
    
    response = tokenizer.decode(gen_out[0][full_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[AI 最终回答]: {response}")
    print(f"[真实标准答案]: {sample['answers']}")

if __name__ == '__main__':
    run_longbench_demo()
