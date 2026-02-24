# 算法核心与开发日志 (Algorithm Development Log)

## 1. 算法核心原理 (Core Theoretical Foundation)
本研究提出一种基于**能量赤字 (Energy Deficit)** 和 **非对称泄放 (Asymmetric Shift)** 的新型长文本 Attention 机制，旨在极低显存占用下实现高效的多文档/长上下文生成。

### 阶段 1：获取局部自由能
将极长的上下文切分为 $N$ 个 Chunks。为每个 Chunk 独立调用 Attention 并提取其局部自由能配分函数 $LSE_i$。

### 阶段 2：动态自适应基线与赤字计算
摒弃全局绝对均值 (Mean Field)，采用当前 Query 下所有 Context 的 Top-K 平均值作为健康基线 $Target\_LSE$。
- **赤字计算**：$\Delta_i = Target\_LSE - LSE_{doc\_i}$
- 若 $\Delta_i \gg 0$，能量缺失，判定为噪音块；若 $\Delta_i \le 0$，包含黄金信号。

### 阶段 3：软性泄放与非对称平移
利用 PyTorch 原生的 $F.softplus$ 执行平滑的非对称惩罚注入：
- **Offset 计算**：$\text{Offset}_i = - \text{F.softplus}(\Delta_i)$
- **非对称注入**：$\widetilde{LSE}_{doc\_i} = LSE_{doc\_i} + \text{Offset}_i$
此优雅的软函数实现了：对确定性噪音施加精确的 1:1 负向线性惩罚，对黄金信号则实现 0 惩罚的无损旁路穿透。
*(极其重要：System Prompt 和 Query 的 LSE 保持初始锚点状态，绝对不能加上任何 offset。)*

### 阶段 4：安全全局归约
在获取了修正后的 LSE 后，使用数值安全的在线 Softmax 合并所有局部的输出矩阵 $O_i$，进行线性叠加。

---

## 2. 工程优化与架构演进 (Engineering & Debugging)

### 痛点 A：原始长文本引擎的显存爆炸 (OOM)
- **发现问题**：原始实验代码在最终的 Query 阶段，简单粗暴地将之前并行的所有文档的 KV Cache 通过 `torch.cat` 直接拼成一个超级张量。这种 $O(L)$ 的连续显存申请在 96GB 的高配显卡上也瞬间导致 OOM。
- **解法一瞥 (Chunk-by-Chunk)**：不再建立庞大的全局 Cache Tensor。让 Query 分别遍历各个独立的 Chunk 提取局部的 $O_i$ 和 $LSE_i$，直接在系统内存中对标量规模的 LSE 执行归约与数学乘加，成功消除拼接瓶颈。

### 痛点 B：“物理删除算法”的灾难性陷阱
- **危险路线**：基于“能量赤字”，我们起初设想将那些计算出高 $\Delta_i$ (纯噪音) 的 Chunk 在推理中彻底 `del` 掉，实现极简显存压缩。
- **论证推翻 (三大致命缺陷)**：
  1. **永久失忆**：当前的“噪音”不可预见是否会成为未来生成的“大海捞针”。物理删除导致不可逆信息缺失。
  2. **Multi-Head 冲突 (众口难调)**：不同 Attention Head 职能不同，简单取均值删除会破坏特定捕捉(如长距关联与语法)的表征能力。
  3. **Attention Sinks (注意力吸盘) 崩溃**：LLM 常常将极高 Attention 分配给最初的几个 Token 以稳定 Softmax 分母（不包含强语义）。物理删除 chunk 首部，将直接引发输出乱码。

### 终极回退决策 (Rollback Decision): 抛弃剪枝，执行纯数学无损归约
在工程探讨期间，我们一度设想过**GPU-CPU 异步分级状态机 (Tiered Cache Architecture)** 以实现对高 $\Delta_i$ 块的动态驱逐。但经过严密的理论反思，我们**主动废弃了该方向**。
1. **风险太大**：长文本生成充满“跳跃性注意力”与“大海捞针”情境。哪怕只有 1% 的误判 eviction，都会带来不可逆的信息断层（永久失忆）。
2. **多头注意力 (Multi-Head) 的众口难调**：如果为了某个 Head 的配分低而牺牲整个 Chunk，会葬送其他专门负责位置信息、长程语法的结构性 Head 的表征基础。
3. **最终选择**：回归**纯数学防线**。不删、不驱逐，所有由于 OOM-Safe 机制拆分后的 Chunk KV Cache 均平分秋色地驻留在显存或 PagedMemory 里。我们仅依赖公式 $\widetilde{LSE} = LSE - F.softplus(\Delta)$，让处于极大“赤字”的噪音块在 Softmax 操作后，其混合权重 $W_i$ 自动衰减至无穷小。
这是一种**完全无损 (Lossless)** 的降噪方法。我们通过数学公式的“软屏蔽”实现了物理剪枝达不到的安全性。
