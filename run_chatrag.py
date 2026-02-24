import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine import ResearchEngine

# --- 核心配置 ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DIVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_chatrag_prompt(instruction, documents, query):
    """
    根据 Llama-3 的官方格式构建输入。
    ChatRAG 的特点是：有一个全局指令(Prefix)，一堆互相独立的文档(Contexts)，和最后的问题(Query)。
    """
    system_prompt = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>\n"
    
    # Prefix: 系统提示词 + 任务指令
    prefix = system_prompt + f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n"
    
    # Query: 问题
    query_str = f"Question: {query}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prefix, documents, query_str

def run_chatrag_demo():
    print("[*] 正在构建 ChatRAG-Bench (多文档检索增强问答) 的模拟测试例...")
    
    # 模拟 ChatRAG 的典型数据：检索到了 5 篇长短不一的文档，其中只有 1 篇包含真正答案
    sample = {
        "instruction": "Read the following retrieved documents and answer the user's question accurately.",
        "documents": [
            "Document 1: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "Document 2: Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.",
            "Document 3: The novel A Hundred Years of Solitude was written by Gabriel Garcia Marquez.", # 这篇无关
            "Document 4: Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.", # 包含黄金信号
            "Document 5: Cellular respiration is the process by which biological fuels are oxidized."
        ],
        "query": "What is Python and what does its design philosophy emphasize?",
        "answers": ["Python is a programming language that emphasizes code readability."]
    }
    
    prefix, documents, query = build_chatrag_prompt(sample['instruction'], sample['documents'], sample['query'])
    
    print(f"[*] 成功加载查阅测试数据! 问题: {sample['query']}")
    print(f"[*] 召回文档数量: {len(documents)}")
    print(f"[*] 初始化模型: {MODEL_ID} on {DIVICE}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16 if DIVICE == "cuda" else torch.float32,
            device_map=DIVICE
        )
        engine = ResearchEngine(model)
    except Exception as e:
        print(f"[!] 模型加载失败 (环境受限): {e}")
        print("[!] 脚本已准备就绪，包含完整的上下文并行+能量数学惩罚逻辑，可直接置于服务器执行。")
        return

    # ==========================
    # 开始执行能量赤字算法 (多文档独立并行处理)
    # ==========================
    
    # STAGE 1: Prefix
    print("[*] 阶段 1: 处理 Prefix (任务指令)...")
    engine.inject('prefix')
    p_ids = tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(DIVICE)
    p_pos = torch.arange(p_ids.shape[1]).unsqueeze(0).to(DIVICE)
    with torch.no_grad():
        out = model(p_ids, position_ids=p_pos, use_cache=True)
    prefix_cache = out.past_key_values

    # STAGE 2: Parallel Contexts (每篇 Document 作为独立 Chunk)
    print(f"[*] 阶段 2: 并行处理 {len(documents)} 篇召回文档...")
    engine.inject('context')
    num_ctx = len(documents)
    batched_pre_cache = []
    for pk, pv, pp in prefix_cache:
        batched_pre_cache.append((pk.repeat(num_ctx, 1, 1, 1), pv.repeat(num_ctx, 1, 1, 1), pp))
        
    c_ids = tokenizer(documents, return_tensors='pt', padding=True, add_special_tokens=False).input_ids.to(DIVICE)
    c_pos = torch.arange(p_ids.shape[1], p_ids.shape[1] + c_ids.shape[1]).unsqueeze(0).to(DIVICE)
    with torch.no_grad():
        out = model(c_ids, position_ids=c_pos, past_key_values=batched_pre_cache, use_cache=True)
    
    # 组装全局 Cache (由于我们的引擎已经改成 OOM-Safe，这里只是准备数据结构)
    global_cache = []
    mask = (c_ids != tokenizer.pad_token_id).flatten()
    for layer_out in out.past_key_values:
        k, v, p = layer_out
        len_pre = p_ids.shape[1]
        k_pre, v_pre, p_pre = k[:1, :, :len_pre], v[:1, :, :len_pre], p[:, :len_pre]
        
        # 将多个并行的 Document 展平，传入引擎内部后会被再次自动分块
        k_ctx = k[:, :, len_pre:].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
        v_ctx = v[:, :, len_pre:].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
        p_ctx = p[:, len_pre:].repeat(num_ctx, 1).flatten()[mask].unsqueeze(0)
        
        global_cache.append((torch.cat([k_pre, k_ctx], dim=2), 
                             torch.cat([v_pre, v_ctx], dim=2), 
                             torch.cat([p_pre, p_ctx], dim=1),
                             num_ctx, len_pre))

    # STAGE 3: Query & 安全提纯
    print("[*] 阶段 3 & 4: 注入 Query，触发能量配分软屏蔽 (OOM-Safe)...")
    # ChatRAG 中噪音文档多，Target LSE 阈值可以调苛刻一点，比如 Top-K 取 1
    engine.inject('query', top_k=1) 
    
    q_ids = tokenizer(query, return_tensors='pt', add_special_tokens=False).input_ids.to(DIVICE)
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
    run_chatrag_demo()
