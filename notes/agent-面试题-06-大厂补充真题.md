# 互联网大厂 AI Agent 面试题补充汇总

> 来源：字节跳动公开面经、GitHub llm_interview_note 仓库、社区真实面经整理  
> 更新日期：2026-07-04  
> 说明：本文是 `agent-面试题-04-大厂真题汇总.md` 的补充，聚焦**字节跳动豆包/火山引擎**、**Transformer 底层原理**、**分布式训练**、**推理加速**等更深度的考察方向

---

## 目录

1. [Transformer 底层原理深挖](#一-transformer-底层原理深挖)（Q1–Q12）
2. [训练技术与大规模分布式](#二-训练技术与大规模分布式)（Q13–Q22）
3. [对齐与微调进阶](#三-对齐与微调进阶)（Q23–Q32）
4. [推理加速与部署优化](#四-推理加速与部署优化)（Q33–Q42）
5. [Agent 架构进阶](#五-agent-架构进阶)（Q43–Q57）
6. [RAG 深度题](#六-rag-深度题)（Q58–Q70）
7. [系统设计大题](#七-系统设计大题)（Q71–Q78）
8. [数学与统计基础](#八-数学与统计基础)（Q79–Q84）
9. [LangGraph 深度](#九-langgraph-深度)（Q85–Q95）
10. [MCP 协议深度](#十-mcp-协议深度)（Q96–Q105）

---

## 一、Transformer 底层原理深挖

> 字节、阿里、百度算法岗必考，工程岗也常出现

### Q1. 【进阶】Self-Attention 的时间和空间复杂度是多少？

**难度**：★★★
**类型**：复杂度分析

**答案**：

- **时间复杂度**：O(n² · d)
  - n 是序列长度，d 是隐层维度
  - QK^T 矩阵乘法：O(n² · d)，对每个 token 都要计算与其他所有 token 的相关度
- **空间复杂度**：O(n²)
  - 需要存储 n×n 的 Attention 矩阵

**工程影响**：序列长度翻倍，计算量变 4 倍。这是长序列场景（128K token）的核心瓶颈，也是 Flash Attention 要解决的问题。

---

### Q2. 【进阶】Flash Attention 的核心思想是什么？解决了什么问题？

**难度**：★★★★
**类型**：工程优化

**答案**：

**问题**：标准 Attention 需要把 n×n 的注意力矩阵写到 HBM（GPU 高带宽内存），再读回来做 softmax，内存 I/O 成为瓶颈（不是计算量本身）。

**Flash Attention 核心思想**：
1. **Tiling（分块）**：把 Q、K、V 分成小块，在 SRAM（快速 on-chip 内存）中计算，避免频繁读写 HBM
2. **Online softmax**：用数值稳定的在线算法，不需要存储完整 n×n 矩阵
3. **重计算（Recomputation）**：反向传播时重新计算 Attention，而不是存储中间结果

**效果**：
- 内存占用从 O(n²) 降为 O(n)
- 速度提升 2–4 倍（主要是 I/O 减少）
- 支持更长序列

---

### Q3. 【进阶】GQA（Grouped Query Attention）是什么？为什么 LLaMA3、Mistral 都用它？

**难度**：★★★★
**类型**：架构改进

**答案**：

**背景**：
- MHA（Multi-Head Attention）：每个 head 有独立 Q/K/V，KV Cache 大
- MQA（Multi-Query Attention）：所有 head 共享一组 K/V，KV Cache 小但质量下降
- GQA（Grouped Query Attention）：折中方案，G 个 head 共享一组 K/V

```
MHA:  [Q1 K1 V1] [Q2 K2 V2] [Q3 K3 V3] [Q4 K4 V4]
MQA:  [Q1       ] [Q2       ] [Q3       ] [Q4       ]
       [    K  V ]
GQA:  [Q1 Q2] [K1 V1]   [Q3 Q4] [K2 V2]
```

**为什么用 GQA**：
- KV Cache 大小减少至 MHA 的 1/G（G=8 时减少 87.5%）
- 推理速度显著提升（KV Cache 读取是推理瓶颈之一）
- 质量接近 MHA，远好于 MQA

---

### Q4. 【进阶】RoPE（旋转位置编码）的原理和优势？

**难度**：★★★★
**类型**：位置编码

**答案**：

**核心思想**：通过旋转矩阵将位置信息融入 Q 和 K，使得 Q_i · K_j 的内积只依赖于相对位置 (i-j)，而不是绝对位置。

**公式**：对向量 x 在位置 m 处，旋转角度 mθ：
```
RoPE(x, m) = R_m · x
其中 R_m 是旋转矩阵，角度与位置 m 相关
```

**三大优势**：
1. **外推性好**：测试时序列长度超过训练长度，性能下降比绝对位置编码少
2. **相对位置感知**：自然编码相对距离信息
3. **无额外参数**：不需要学习位置 embedding

**扩展长度的方法**：YaRN、LongRoPE 等插值方法，让 LLaMA 2（4K context）扩展到 100K+。

---

### Q5. 【中级】Layer Norm 放在 Attention 之前（Pre-Norm）还是之后（Post-Norm）有什么区别？

**难度**：★★★
**类型**：训练稳定性

**答案**：

| 特性 | Post-Norm（原始 Transformer） | Pre-Norm（现代大模型） |
|------|-------------------------------|----------------------|
| 训练稳定性 | 较差，深层网络易梯度消失 | 较好，梯度流更稳定 |
| 最终性能 | 稍好（充分训练后） | 稍差 |
| 学习率要求 | 需要 warmup，学习率不能太大 | 可以用更大学习率 |
| 工业使用 | 早期 BERT | GPT-3、LLaMA 等现代模型 |

**现代大模型普遍用 Pre-Norm** 是因为训练稳定性更重要——万亿 token 训练，中途崩溃代价太大。

---

### Q6. 【进阶】什么是 Sparse Attention？有哪些实现方式？

**难度**：★★★★
**类型**：效率优化

**答案**：

标准 Attention 是 dense（每个 token 都关注所有其他 token），Sparse Attention 只关注部分 token，降低计算量。

**主要实现方式**：
1. **Local Attention**：只关注附近 w 个 token（滑动窗口）
2. **Strided Attention**：每隔 k 个 token 关注一次（跳跃模式）
3. **Global + Local**：少数 Global token（如 [CLS]）关注所有，其余只关注局部
4. **BigBird / Longformer**：结合 Local + Global + Random 三种模式

**代表模型**：Longformer（长文档），BigBird（长序列）。

---

### Q7. 【中级】为什么 Softmax 在计算 Attention 时要做数值稳定处理？

**难度**：★★
**类型**：数值计算

**答案**：

**问题**：`softmax(x_i) = exp(x_i) / Σ exp(x_j)`，当 x 很大时 `exp(x)` 会溢出（float32 约在 x > 88 时溢出）。

**标准处理**：减去最大值后再算：
```python
# 数值不稳定版本
scores = exp(Q @ K.T / sqrt(d_k))

# 数值稳定版本
m = max(Q @ K.T / sqrt(d_k))
scores = exp(Q @ K.T / sqrt(d_k) - m)  # 减去最大值
```

**为什么减最大值不影响结果**：
```
softmax(x - m)_i = exp(x_i - m) / Σ exp(x_j - m)
                 = exp(x_i) / Σ exp(x_j)  # 分子分母同除 exp(m)，结果不变
```

---

### Q8. 【进阶】KV Cache 的显存占用如何估算？

**难度**：★★★
**类型**：工程计算

**答案**：

```
KV Cache 大小 = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element

以 LLaMA-3 8B（BF16）为例：
- num_layers = 32
- num_kv_heads = 8（GQA，原来 32 个 head 共享 8 组 KV）
- head_dim = 128
- seq_len = 8192
- bytes = 2（BF16）

KV Cache = 2 × 32 × 8 × 128 × 8192 × 2 = 1,073,741,824 bytes ≈ 1 GB

batch_size=32 时 = 32 GB
```

**工程意义**：这就是为什么长上下文场景显存容易爆，以及为什么要用 PagedAttention（vLLM）或 GQA 来节省 KV Cache。

---

### Q9–Q12. 其他底层原理高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q9 | 什么是 ALiBi 位置编码？ | 在 Attention 分数上加线性偏置，无需位置 embedding，外推性强 |
| Q10 | Decoder-only 为什么比 Encoder-Decoder 在大模型时代更流行？ | KV Cache 利用效率更高；zero-shot 更强；训练目标更统一 |
| Q11 | 什么是因果掩码（Causal Mask）？为什么 GPT 需要它？ | 下三角矩阵：token 只能看到自己和之前的 token，保证自回归生成 |
| Q12 | Softmax 在 Attention 中出现 "attention sink" 是什么现象？ | 第一个 token 被分配异常高的注意力权重；StreamingLLM 利用此特性做长序列推理 |

---

## 二、训练技术与大规模分布式

### Q13. 【进阶】ZeRO Stage 1/2/3 各优化了什么？

**难度**：★★★★
**类型**：分布式训练

**答案**：

ZeRO（Zero Redundancy Optimizer）将模型状态分片到多个 GPU：

| Stage | 分片内容 | 显存节省 | 通信量 |
|-------|---------|---------|--------|
| Stage 1 | 优化器状态（Adam 的 m/v/param） | 4× | 略增 |
| Stage 2 | 优化器状态 + 梯度 | 8× | 增加 |
| Stage 3 | 优化器状态 + 梯度 + 模型参数 | 64×（理论） | 大幅增加 |

**选择原则**：
- 单 GPU 放得下模型 → 不用 ZeRO，或 Stage 1 降低显存峰值
- 模型放不下单 GPU → Stage 3（配合 DeepSpeed）
- 追求吞吐量 → Stage 2（通信量适中）

---

### Q14. 【进阶】数据并行、张量并行、流水线并行分别解决什么问题？

**难度**：★★★★
**类型**：分布式策略

| 策略 | 解决的问题 | 通信瓶颈 | 典型工具 |
|------|----------|---------|---------|
| 数据并行（DP） | 提升吞吐量 | AllReduce 梯度同步 | DDP、ZeRO |
| 张量并行（TP） | 单层参数太大，放不进单 GPU | AllReduce（每层） | Megatron-LM |
| 流水线并行（PP） | 层数太多，GPU 内存不够 | P2P（层间传激活） | GPipe、PipeDream |

**3D 并行**（DP + TP + PP）：用于千亿参数模型训练（如 Megatron-Turing NLG 530B）。

---

### Q15. 【中级】Gradient Checkpointing 如何权衡显存和计算时间？

**难度**：★★★
**类型**：训练优化

**答案**：

**问题**：反向传播需要前向传播的中间激活值（每层输出），全部存储显存压力大。

**Gradient Checkpointing 原理**：
- 只保存部分"检查点"层的激活值
- 反向传播到某层时，从最近的检查点重新前向计算获得所需激活值

**权衡**：
- 显存节省约 √n 倍（n 为层数）
- 计算量增加约 1/3（每层平均被计算约 1.33 次）

**实践**：通常在显存紧张时开启，训练速度降低约 20–30% 但能训练更大 batch 或更大模型。

---

### Q16. 【进阶】为什么预训练数据要去重？MinHash 去重是怎么工作的？

**难度**：★★★
**类型**：数据处理

**答案**：

**为什么要去重**：
- 重复数据导致模型过度记忆特定文本（memorization），泛化性差
- 重复内容会"浪费"训练 token，降低数据效率
- 可能导致模型直接复现训练数据（隐私/版权风险）

**MinHash LSH 去重原理**：
1. 对每篇文档做 n-gram 分词（通常 5-gram）
2. 用 K 个哈希函数计算 MinHash 签名（长度 K 的向量）
3. 用 LSH（局部敏感哈希）把相似 MinHash 签名分到同一个 bucket
4. 只在同 bucket 内精细比较 Jaccard 相似度
5. 相似度超过阈值（通常 0.8）则删除其中一篇

**效果**：可在 O(n log n) 复杂度内处理百亿级文档去重。

---

### Q17–Q22. 其他训练深度题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q17 | 混合精度训练（BF16 vs FP16）的区别？ | BF16 指数位更多（8位），不易溢出，更适合大模型；FP16 精度更高但容易数值不稳定 |
| Q18 | 什么是 Token Dropping？ | 训练时随机丢弃部分 token 减少计算量，类似 Dropout；Switch Transformer 用于 MoE |
| Q19 | MoE 的负载均衡如何实现？ | 辅助 loss（auxiliary loss）惩罚 expert 使用不均匀，防止所有输入都路由到同一个 expert |
| Q20 | 什么是 Curriculum Learning？在大模型训练中怎么用？ | 先训容易数据，再训困难数据；实践中常用于从短序列过渡到长序列训练 |
| Q21 | 训练中 loss spike（损失突增）如何处理？ | 回滚到之前 checkpoint，降低学习率，检查数据质量（是否有脏数据批次）|
| Q22 | 什么是 Compute-Optimal Training？Chinchilla 定律说了什么？ | 参数量 N 和训练 token 数 D 的最优比约为 D ≈ 20N；之前 GPT-3 等模型 token 数不足 |

---

## 三、对齐与微调进阶

### Q23. 【进阶】DPO 和 PPO 的核心区别？DPO 为什么不需要奖励模型？

**难度**：★★★★
**类型**：对齐方法

**答案**：

**PPO（RLHF）流程**：
```
偏好数据 → 训练 RM → RM 在线打分 → PPO 优化策略
需要：策略模型（Actor）+ 值函数模型（Critic）+ 奖励模型（RM）+ 参考模型
```

**DPO（Direct Preference Optimization）**：
```
偏好数据 → 直接优化 → 策略模型
需要：策略模型 + 参考模型（冻结）
```

**DPO 的数学本质**：将 RLHF 的 RL 目标（最大化奖励 - KL 散度）闭式求解，得到等价的分类 loss：
```
L_DPO = -log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))
```
其中 y_w 是被偏好的回复，y_l 是不被偏好的回复。

**优缺点对比**：

| | PPO | DPO |
|--|-----|-----|
| 训练复杂度 | 高（4 个模型同时运行） | 低（2 个模型）|
| 训练稳定性 | 较差（RL 本身不稳定） | 较好 |
| 需要在线采样 | 是（持续探索） | 否（离线数据）|
| 效果 | 通常更好（特别是复杂任务） | 接近 PPO，数学/代码稍差 |

---

### Q24. 【进阶】LoRA 的 rank 如何选择？什么时候用 LoRA，什么时候全参数微调？

**难度**：★★★
**类型**：微调策略

**答案**：

**rank 选择原则**：
- rank=4–8：轻量适配（改变输出风格、格式）
- rank=16–32：中等任务（特定领域知识注入）
- rank=64–128：复杂任务（接近全参数微调效果）

**选 LoRA 的条件**：
- 计算资源受限
- 需要保留原始模型能力（只是增强特定功能）
- 需要快速切换多个任务（多个 LoRA adapter 轮换）
- 部署时显存受限（LoRA 推理只需加载小 adapter）

**选全参数微调的条件**：
- 目标分布与预训练差异极大
- 需要彻底改变模型行为
- 有足够算力和高质量数据

---

### Q25. 【进阶】QLoRA 比 LoRA 多了什么？NF4 量化是什么？

**难度**：★★★★
**类型**：量化微调

**答案**：

**QLoRA = 4-bit 量化基础模型 + LoRA 适配器**：

1. **NF4（Normal Float 4）量化**：专门为正态分布的神经网络权重设计的 4bit 数据类型，量化误差比普通 INT4 小约 10%
2. **双重量化（Double Quantization）**：对量化的缩放因子再量化，进一步节省显存（约 0.37bits/param）
3. **分页优化器（Paged Optimizer）**：用 CPU 内存缓冲 GPU 显存峰值，防止 OOM

**效果**：65B 参数模型（通常需要 ~130GB GPU 内存）可以在单张 48GB A100 上微调，质量接近全精度 LoRA。

---

### Q26–Q32. 其他微调对齐高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q26 | 什么是 RLAIF？和 RLHF 的区别？ | 用 AI 模型（如 Claude）替代人类标注偏好，成本低但引入 AI 偏见 |
| Q27 | 灾难性遗忘如何缓解？ | 混入通用数据、EWC 正则化、LoRA（少改参数）、Replay Buffer |
| Q28 | SFT 数据格式是什么？质量和数量哪个更重要？ | Instruction-Response 对；质量更重要，1000条高质量 > 100万低质量 |
| Q29 | 什么是 Reward Hacking？如何缓解？ | 模型学会骗过奖励模型而不是真正改善；解决：改进 RM、多样化奖励信号 |
| Q30 | Proximal Policy Optimization（PPO）的 clip 机制是什么？ | 将策略更新比例 clip 到 [1-ε, 1+ε]，防止一步更新太大导致训练崩溃 |
| Q31 | ORPO 是什么？相比 DPO 有什么改进？ | Odds Ratio Preference Optimization，无需参考模型，单步训练 SFT + 对齐 |
| Q32 | 什么是 Constitutional AI 中的"宪法"？ | 一组明确的原则列表（如"不帮助非法活动"），用于 AI 自我批评和修正 |

---

## 四、推理加速与部署优化

### Q33. 【进阶】vLLM 的 PagedAttention 核心思想？

**难度**：★★★★
**类型**：推理系统

**答案**：

**问题**：传统 KV Cache 为每个请求预分配连续内存（最大序列长度），但实际生成长度不定，导致大量内存碎片化和浪费（平均利用率只有 20–40%）。

**PagedAttention 思路**（借鉴 OS 虚拟内存分页）：
1. 把 KV Cache 分成固定大小的 **物理块**（如每块 16 个 token）
2. 每个序列有 **逻辑块** 到 **物理块** 的映射表
3. 物理块可以不连续，按需分配
4. 支持 **Copy-on-Write**：多个请求共享相同前缀的 KV Cache（用于 Beam Search 和共享系统提示）

**效果**：内存利用率从 20–40% 提升到 90%+，吞吐量提升 2–4 倍。

---

### Q34. 【进阶】Continuous Batching（连续批处理）是什么？

**难度**：★★★
**类型**：推理吞吐

**答案**：

**Static Batching 问题**：一个 batch 中所有请求必须等最长的那个请求完成才能出结果，短请求的 GPU 时间被浪费。

**Continuous Batching（迭代级调度）**：
- 每生成一个 token 后，检查哪些请求已完成（生成了 EOS）
- 立即将完成的请求移出 batch，新请求填入
- GPU 始终保持满负载

**效果**：延迟降低，吞吐量提升 3–10 倍（取决于请求长度分布）。vLLM、TGI 都实现了 Continuous Batching。

---

### Q35. 【进阶】Speculative Decoding 原理和 acceptance rate 如何影响加速比？

**难度**：★★★★
**类型**：推理加速

**答案**：

**原理**：
1. 小（草稿）模型快速生成 k 个候选 token
2. 大（目标）模型并行验证这 k 个 token（一次前向传播）
3. 从左到右接受：如果某个 token 与大模型预测一致则接受，否则拒绝并用大模型的预测替换

**加速比公式**：
```
加速比 ≈ (1 + k × α) / (1 + γ)
其中：α = acceptance rate（每个 token 被接受的概率）
      γ = 小模型 / 大模型的计算开销比
```

**α 的影响**：
- α = 1（全部接受）：加速比接近 k+1 倍
- α = 0（全部拒绝）：无加速，反而有开销
- 实际上 α ≈ 0.7–0.9（相同领域数据），加速比约 2–3 倍

**适用场景**：代码生成（确定性强，α 高）；不适合高创意性写作（α 低）。

---

### Q36–Q42. 其他推理优化高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q36 | TTFT 和 TPOT 分别是什么？如何分别优化？ | TTFT（首 token 延迟）→ 优化 prefill；TPOT（每 token 延迟）→ 优化 decode，用 GQA/MQA |
| Q37 | INT4/INT8 量化对精度的典型影响？ | INT8 通常精度损失 < 1%；INT4 约 1–3%，但复杂推理任务可能下降 10%+ |
| Q38 | PTQ 和 QAT 的区别？大模型为什么主要用 PTQ？ | PTQ 无需重训，成本低；大模型重训代价太高，GPTQ/AWQ 等 PTQ 方法已足够好 |
| Q39 | Tensor Parallelism 的通信瓶颈在哪里？ | 每层 Attention 和 FFN 都需要 AllReduce，层数越多通信越多 |
| Q40 | 什么是 Chunked Prefill？ | 把长 Prompt 的 prefill 分块处理，避免阻塞 decode 请求，降低 TTFT 方差 |
| Q41 | 量化 Embedding 层有什么特殊挑战？ | Embedding 层的权重分布不是正态的，用 NF4 效果差，通常保持 FP16 |
| Q42 | 什么是 KV Cache 压缩？H2O 方法是什么？ | 只保留高注意力权重的 KV（heavy hitter），其余丢弃；牺牲少量精度大幅节省显存 |

---

## 五、Agent 架构进阶

### Q43. 【进阶】Chain-of-Thought、Tree-of-Thought、Graph-of-Thought 的区别？

**难度**：★★★
**类型**：推理框架对比

**答案**：

| 框架 | 结构 | 适用场景 | 局限 |
|------|------|---------|------|
| CoT（思维链） | 线性推理步骤 | 数学、逻辑 | 一条路错则全错 |
| CoT-SC（自洽） | 多条线性路径 + 投票 | 有唯一正确答案 | token 消耗 N 倍 |
| ToT（思维树） | 树状搜索，剪枝无效路径 | 需要探索+回溯的问题 | 需要评估函数 |
| GoT（思维图） | 有向图，支持路径合并 | 需要 Merge 结果的问题 | 最复杂，实现难 |

**字节/百度追问**：ToT 如何决定哪些分支要剪枝？
→ 用 LLM 作为"价值评估函数"对每个中间状态打分（好/中/差），只保留好的分支继续扩展。

---

### Q44. 【进阶】如何设计一个能自我纠错的 Agent？

**难度**：★★★★
**类型**：系统设计

**答案**：

**Reflexion 架构**（Shinn et al., 2023）：

```
执行 → 评估结果 → 生成语言反思 → 存入 Episodic Memory → 下次执行时参考
```

**关键设计**：
1. **评估器（Evaluator）**：判断执行是否成功（代码通过测试、任务完成标准）
2. **反思生成器（Reflector）**：生成"为什么失败"的语言总结
3. **记忆存储**：反思以文本形式存储在 context 中（短期）或向量数据库（长期）
4. **最大尝试次数**：防止无限循环（通常 3–5 次）

**代码示例**：
```python
for attempt in range(max_attempts):
    result = agent.execute(task)
    if evaluator.is_success(result):
        return result
    reflection = reflector.reflect(task, result, "为什么失败了？下次如何改进？")
    memory.add(reflection)
    task_with_memory = f"历史反思：{memory.get_all()}\n\n当前任务：{task}"
```

---

### Q45. 【进阶】如何处理 Agent 的长期规划（Long-Horizon Planning）挑战？

**难度**：★★★★
**类型**：规划架构

**答案**：

**核心挑战**：
1. 任务步骤多，早期错误会级联放大
2. 中途状态难以评估
3. 上下文窗口不够用

**应对方案**：

1. **分层规划（Hierarchical Planning）**：
   - 高层 Agent：生成粗粒度计划（5–10 个大步骤）
   - 低层 Agent：执行每个大步骤的细节
   
2. **计划 + 执行 + 修正循环**：
   ```
   Plan → Execute Step 1 → Observe → (Re-plan if needed) → Execute Step 2 → ...
   ```
   
3. **里程碑检查点**：每完成一个阶段，验证是否达到预期中间状态

4. **外部规划工具**：对于格式固定的任务（如代码项目），用 Project Manager 工具维护 TODO 列表，而不是靠模型记忆

---

### Q46. 【进阶】Agent 系统中如何实现可靠的工具调用重试机制？

**难度**：★★★
**类型**：工程健壮性

**答案**：

```python
import asyncio
from enum import Enum

class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR = "linear"
    IMMEDIATE = "immediate"

async def execute_with_retry(
    tool_name: str,
    tool_input: dict,
    max_retries: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
) -> str:
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            result = await tool_registry.execute(tool_name, tool_input)
            return result
        except TransientError as e:
            # 可重试的错误（网络超时、限流）
            last_error = e
            if attempt < max_retries:
                if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                    wait = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                else:
                    wait = 1.0
                await asyncio.sleep(wait)
        except PermanentError as e:
            # 不可重试的错误（参数错误、工具不存在）
            return f"Error: {e}"  # 直接返回错误给模型
    
    return f"Error: Tool failed after {max_retries} retries: {last_error}"
```

**关键原则**：
- 可重试错误（网络、限流）→ 指数退避重试
- 不可重试错误（参数错误）→ 立即返回错误给 LLM 让其修正
- 超时错误 → 作为 tool_result 返回，不要在应用层 raise

---

### Q47–Q57. 其他 Agent 进阶高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q47 | 如何评估 Agent 效果？任务完成率之外还看什么？ | 步骤效率（完成任务的平均步骤数）、工具使用精准率、错误恢复率、人工干预率 |
| Q48 | Agent 系统如何做 A/B 测试？ | 按用户/请求分流；指标需等待足够样本；注意长尾任务的统计显著性 |
| Q49 | 什么是 Agent 的"幻觉级联"？ | 早期工具调用幻觉导致后续步骤全部基于错误信息；通过工具结果验证步骤阻断 |
| Q50 | 如何让 Agent 主动向用户澄清歧义？ | Prompt 中声明"遇到关键信息缺失时主动提问"；设计 ask_user 工具 |
| Q51 | 什么是 Agent 的"幂等会话"？ | 相同任务重新触发，Agent 能检测到已完成并返回缓存结果 |
| Q52 | Computer Use Agent 的主要安全风险？ | 截图含敏感信息；鼠标/键盘操作可能误触；间接 Prompt Injection（页面上的恶意文本）|
| Q53 | 如何设计 Agent 的权限分级系统？ | 只读权限（查询）/ 写入权限（修改）/ 执行权限（删除/发送）；高危操作需人工审批 |
| Q54 | 多模态 Agent 如何处理图像 token 数量？ | 图像压缩 + 低分辨率预览 + 按需高清；LLaVA 等用视觉编码器降维 |
| Q55 | 什么是 Agent Protocol？ | AgentOps 等社区制定的 Agent 标准化接口，类似 MCP 但面向 Agent 间通信 |
| Q56 | Agent 如何处理"不可能完成"的任务？ | 识别后主动声明任务不可行并解释原因，而非无限循环尝试 |
| Q57 | 如何设计一个代码助手 Agent 的错误反馈循环？ | 执行代码 → 捕获 stderr/traceback → 作为 tool_result → 模型分析 → 修复 → 重试（最多 N 次）|

---

## 六、RAG 深度题

### Q58. 【进阶】什么是 GraphRAG？和传统 RAG 有什么本质区别？

**难度**：★★★★
**类型**：新型 RAG

**答案**：

**传统 RAG 的局限**：只做向量相似度匹配，无法回答需要**跨文档推理**的问题（如"A 公司和 B 公司有什么关联？"）。

**GraphRAG（Microsoft 2024）**：
1. 用 LLM 从文档中提取**实体**（人/公司/地点）和**关系**（A 收购了 B）
2. 构建**知识图谱**（Knowledge Graph）
3. 查询时：在图谱中找相关实体 → 多跳推理找关联 → 聚合信息生成答案

**两种查询模式**：
- **Local Search**：围绕特定实体的详细信息查询
- **Global Search**：跨整个语料库的全局摘要性问题

**适用场景**：企业知识库、法律文档、学术文献（需要跨文档推理的场景）。

---

### Q59. 【进阶】什么是 CRAG（Corrective RAG）？

**难度**：★★★★
**类型**：自适应 RAG

**答案**：

CRAG 在标准 RAG 流程中加入**检索质量评估**步骤：

```
Query → 检索 Top-K → 评估相关性分数
                        ↓
          相关（high） → 直接用检索结果
          模糊（low）  → 对 chunk 做精细提取，过滤噪声
          不相关        → 丢弃检索结果，触发 Web Search 补充
                        ↓
                     生成最终答案
```

**评估方法**：用 LLM 作 relevance 评分器（0–1），或训练专门的 cross-encoder 分类器。

**优势**：比 Naive RAG 更鲁棒，在检索质量差时自动降级到 Web Search。

---

### Q60. 【进阶】Self-Query Retriever 是什么？

**难度**：★★★
**类型**：查询优化

**答案**：

普通向量检索只做语义搜索，无法利用文档的**结构化元数据**（如日期、作者、类别）。

**Self-Query Retriever**：让 LLM 自动从自然语言 Query 中提取**语义部分**和**元数据过滤条件**：

```
用户问："2024年以后关于量化的论文"
                ↓ LLM 解析
语义搜索：  "量化" → 向量检索
元数据过滤：year > 2024 → 向量库 filter
```

**实现**：在向量数据库（Pinecone、Qdrant）中，向量搜索同时附带元数据过滤（pre-filtering 或 post-filtering）。

---

### Q61–Q70. 其他 RAG 进阶高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q61 | Multi-Vector Retrieval 是什么？ | 一个文档生成多个向量（摘要向量 + 子问题向量 + 原文向量），提高召回覆盖率 |
| Q62 | 如何处理 RAG 中的表格数据？ | 表格转为文字描述 + 结构化存储；或直接用 SQL 查询代替向量检索 |
| Q63 | Embedding 模型如何选型？ | 维度（越高越准但越慢）、语言支持、领域适配、开源 vs 商业、BGE vs OpenAI vs Cohere |
| Q64 | 增量更新向量数据库的挑战？ | Embedding 模型更新导致旧向量失效；需要分批重索引 |
| Q65 | RAG 中 context window 放多少 chunk 合适？ | chunk × chunk_size < 0.5 × context_window；太多会稀释相关性 |
| Q66 | 什么是 Sentence Window Retrieval？ | 以句子粒度建索引，检索时扩展返回上下几句，兼顾精度和上下文 |
| Q67 | 如何评估 Chunking 策略是否合理？ | 检索命中率（Recall@K）+ 人工抽样检查 chunk 是否完整 |
| Q68 | RAG 和 Long Context LLM 如何选择？ | 文档量大/动态更新 → RAG；文档少/固定/需深度推理 → Long Context |
| Q69 | 什么是 Dense Passage Retrieval（DPR）？ | Facebook 的双塔检索模型，Query 和 Document 分别编码，内积做相似度 |
| Q70 | RAG 系统如何做缓存优化？ | Semantic Cache：相似 Query 复用检索结果；TTL 控制缓存有效期 |

---

## 七、系统设计大题

> 字节、阿里常考的开放性系统设计题，需要15–20分钟完整描述架构

### Q71. 【系统设计】设计一个 LLM 驱动的客服系统

**设计要点**：

```
┌─────────────────────────────────────────────────────┐
│                  客服 Agent 系统架构                   │
├─────────────┬──────────────────┬────────────────────┤
│  输入处理层  │    核心 Agent 层  │    工具/知识层       │
│ ─────────── │ ──────────────── │ ────────────────── │
│ 意图识别     │  RAG 知识检索    │  产品知识库         │
│ 敏感词过滤   │  多轮对话管理    │  工单系统 API       │
│ 语言检测     │  Tool Calling   │  订单查询 API       │
│             │  人工转接判断    │  退款流程工具        │
└─────────────┴──────────────────┴────────────────────┘
```

**关键设计决策**：
1. **意图分类**：先用轻量模型分类（咨询/投诉/退款/转人工），再路由到对应 Agent
2. **范围约束**：System Prompt 中列举白名单话题，超出范围优雅拒绝并转人工
3. **多轮记忆**：用 session_id 关联历史对话，关键信息（订单号、用户诉求）结构化存储
4. **人工转接触发**：情绪检测（负面情绪 > 阈值）/ 重复问题（3次未解决）/ 复杂问题
5. **可观测性**：每轮对话记录 trace，人工审核抽样，持续优化 Prompt

---

### Q72. 【系统设计】设计豆包的个性化记忆系统

**设计要点**：

```
用户对话
    ↓
记忆提取器（LLM）
    ↓
提取三类记忆：
├── 用户偏好（"喜欢简洁回答"、"是程序员"）
├── 历史事件（"上次聊了Python项目"）
└── 明确陈述（"我叫张三"、"我住北京"）
    ↓
存储策略：
├── 向量数据库（语义记忆，可检索）
├── 结构化存储（用户画像，精确查询）
└── TTL 机制（过时信息自动失效）
    ↓
新对话时：根据当前 Query 检索相关记忆 → 注入 System Prompt
```

**隐私设计**：
- 用户可查看/删除自己的记忆
- 敏感信息（密码、银行卡）不存入记忆
- 记忆加密存储，用户数据隔离

---

### Q73. 【系统设计】设计一个代码助手 Agent

**设计要点**：

```
用户需求 → 需求理解 → 生成代码 → 执行代码（沙箱） → 检查结果
                                       ↓
                                   失败？
                                   ↓ yes
                             分析错误（traceback）
                             → 修复代码 → 重新执行（最多 N 次）
                                       ↓ success
                                   返回最终代码 + 解释
```

**核心工具**：
- `execute_code(code: str) → {stdout, stderr, exit_code}`（Docker 沙箱）
- `read_file / write_file`
- `search_docs(query: str)`（代码库文档搜索）
- `run_tests(test_file: str)`

**沙箱安全**：网络隔离、文件系统只读（除 /tmp）、执行时间限制 30s、内存限制 512MB。

---

### Q74. 【系统设计】设计一个实时幻觉检测系统

**设计要点**：

```
LLM 输出
    ↓
【多路验证】
├── 事实核查：提取声明 → RAG 检索依据 → 一致性判断
├── 自我一致性：低温度重采样 3 次 → 主要结论是否一致
└── 引用验证：识别引用 → 检查来源是否真实存在
    ↓
置信度打分（0–1）
    ↓
高置信 → 直接输出
低置信 → 标注"以下内容可能不准确" / 触发人工审核
```

**延迟控制**：
- 事实核查并行进行（不串行等待）
- 简单问题跳过验证（意图分类：闲聊/创作类不需要事实核查）

---

### Q75–Q78. 其他系统设计题要点

| 题号 | 题目 | 核心设计要点 |
|------|------|------------|
| Q75 | 设计多模态 Agent（图片理解）| 视觉编码器 → 图像 token → LLM；token 压缩（256 vs 1024 token/图）|
| Q76 | 设计企业知识库 RAG 系统 | 权限控制（用户只能检索有权限的文档）+ 多租户隔离 + 增量索引 |
| Q77 | 设计 AI 写作助手的长文档协作 | 分段生成 + 大纲先行 + 章节间一致性检查 + 版本管理 |
| Q78 | 设计高并发 LLM 推理服务 | 请求队列 + 优先级调度 + Dynamic Batching + 限流降级 + 多副本负载均衡 |

---

## 八、数学与统计基础

### Q79. Cross-Entropy Loss 为什么适合语言模型？和 Perplexity 的关系？

**答案**：

语言模型是**多分类问题**（每个位置从词表大小的类别中选一个），Cross-Entropy 是多分类的标准 loss：
```
L = -Σ y_i × log(p_i)  ≈  -log(p_correct_token)
```

**Perplexity 的关系**：
```
PPL = exp(L)  （L 是平均 token 的 Cross-Entropy loss）
```
PPL=100 意味着模型平均在"100选1"中选择下一个 token，数值越低越好。

---

### Q80. KL 散度的非对称性在 RLHF 中意味着什么？

**答案**：

```
KL(P||Q) ≠ KL(Q||P)

RLHF 用的是 KL(π_θ || π_ref)（正向 KL）
```

**正向 KL（mode-seeking）**：要求策略 π_θ 在 π_ref 有高概率的地方也要有高概率，防止策略漂移太远。

**为什么不用反向 KL**：反向 KL 是 mean-seeking，会导致策略趋于"平均"，失去生成多样性。

---

### Q81–Q84. 其他数学基础

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q81 | Softmax 的温度系数作用？ | `softmax(x/T)`，T→0 变 argmax，T→∞ 变均匀分布 |
| Q82 | 为什么用 Log Softmax 而不是 Softmax？ | 数值稳定（避免 log(very_small_number)），计算更高效 |
| Q83 | Attention 的 Scaled Dot-Product 为什么能工作？ | Q·K 做内积度量相似度，softmax 归一化为概率，V 加权求和 |
| Q84 | 什么是 Lipschitz 连续性？和神经网络训练有什么关系？ | 梯度上界，Spectral Normalization 用于保证 GAN 判别器稳定 |

---

## 九、LangGraph 深度

### Q85. 【进阶】LangGraph 的 Send API 是什么？什么时候用？

**难度**：★★★★
**类型**：框架高级特性

**答案**：

**Send API** 用于从一个 node 动态创建多个并发分支（Map-Reduce 模式）：

```python
from langgraph.types import Send

def plan_node(state: State) -> list[Send]:
    tasks = state["task_list"]
    # 动态为每个任务创建一个并发 agent
    return [Send("worker_node", {"task": task}) for task in tasks]

# 区别于 add_conditional_edges：
# conditional_edges：路由到固定的下一个 node
# Send：动态创建 N 个并发 worker，每个有自己的 state
```

**适用场景**：
- 一个规划节点拆分出多个并行执行的子任务
- 每个子任务的结果需要聚合到 reduce 节点

---

### Q86. 【进阶】LangGraph 的 Store 和 Checkpointer 有什么区别？

**难度**：★★★
**类型**：状态管理

**答案**：

| 特性 | Checkpointer | Store |
|------|-------------|-------|
| 存储内容 | 每步 StateGraph 的完整状态快照 | 跨会话的长期记忆/数据 |
| 范围 | 单个 thread_id 内 | 跨多个 thread 共享 |
| 用途 | 断点续跑、时光旅行、Human-in-the-Loop | 用户偏好、知识库、共享配置 |
| 访问方式 | 自动（框架管理） | 手动（在 node 中读写）|

**Checkpointer**：每次 `graph.invoke()` 的每步都保存状态，类似数据库的 WAL 日志。

**Store**：类似持久化 key-value 存储，node 主动调用 `store.get/put`。

---

### Q87–Q95. 其他 LangGraph 高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q87 | LangGraph 如何实现并行节点执行？ | 多个 node 从同一个前驱节点出发（多个 add_edge 到不同 node），框架自动并行 |
| Q88 | interrupt_before 和 interrupt_after 的区别？ | before：执行前中断（审批）；after：执行后中断（结果审查）|
| Q89 | LangGraph 的 Subgraph 如何通信？ | 通过共享 State 字段；父图调用子图时传入 state，子图返回 Partial State |
| Q90 | 什么是 LangGraph Platform？ | Hosted API，支持 Cloud Deploy、Cron、Webhook 触发、内置持久化 |
| Q91 | LCEL 的 .with_fallbacks() 如何工作？ | 主 Runnable 失败时自动切换到备用 Runnable（如主模型失败切换到备用模型）|
| Q92 | LangGraph 如何处理 node 执行超时？ | 在 node 函数内部用 asyncio.wait_for；框架层目前无原生超时配置 |
| Q93 | StateGraph 的 START 和 END 是什么？ | 虚拟节点，标记图的入口和出口，不执行实际逻辑 |
| Q94 | 如何在 LangGraph 中实现 Map-Reduce 模式？ | 用 Send API 动态分发 + Annotated list 字段自动聚合各 worker 结果 |
| Q95 | LangGraph 的测试如何做？ | 用 MemorySaver 做 unit test；mock tool 执行器；验证 state 变化路径 |

---

## 十、MCP 协议深度

### Q96. 【进阶】MCP 的 Sampling（采样）能力是什么？

**难度**：★★★★
**类型**：MCP 高级特性

**答案**：

MCP 除了 Tools/Resources/Prompts 三类基础能力，还有 **Sampling（采样）**：

**Server 发起 LLM 调用**：MCP Server 可以通过 Client 请求 Host 调用 LLM，而不只是被 LLM 调用。

```
Server → Client → Host（含 LLM）→ LLM 采样 → 结果返回 Server
```

**典型用途**：
- MCP Server 需要用 LLM 辅助决策（如代码生成 Server 需要 LLM 辅助修改）
- 创建可以自主调用 LLM 的复杂 Server 工具链

**安全考虑**：人工审核采样请求（Human in the Loop），防止递归调用失控。

---

### Q97. 【进阶】MCP 的 Roots 概念是什么？

**难度**：★★★
**类型**：MCP 安全隔离

**答案**：

**Roots**：Client 告知 Server "你可以操作的文件系统根目录范围"，实现权限隔离。

```json
// Client 在初始化时声明 roots
{
  "roots": [
    {"uri": "file:///home/user/project", "name": "My Project"},
    {"uri": "file:///home/user/docs", "name": "Documentation"}
  ]
}
```

**作用**：Server 知道自己的操作边界，不应访问 roots 之外的路径；Client 可以基于 roots 做访问控制检查。

---

### Q98–Q105. 其他 MCP 深度高频题

| 题号 | 题目 | 核心要点 |
|------|------|---------|
| Q98 | MCP 的 JSON-RPC 2.0 消息格式是什么？ | request/response/notification 三种消息类型；必须有 jsonrpc: "2.0" 字段 |
| Q99 | MCP Server 如何做错误处理？ | 返回 JSON-RPC error 对象（code + message + data）；工具执行失败用 isError: true |
| Q100 | stdio 传输下 MCP Server 如何启动？ | Host 用子进程启动 Server，stdin/stdout 作为传输通道 |
| Q101 | 多个 MCP Server 的 Tool 命名冲突如何处理？ | Client 为工具名加 server 前缀（如 `filesystem__read_file`）|
| Q102 | MCP 和 OpenAI Plugin 有什么区别？ | MCP 是本地进程协议，Plugin 是远程 HTTP；MCP 更轻量、安全性更高 |
| Q103 | MCP 如何支持流式响应？ | 通过 SSE（Server-Sent Events）实现长时间运行工具的进度推送 |
| Q104 | 生产环境 MCP Server 如何做认证？ | HTTP+SSE 传输时可加 Bearer Token；stdio 传输依赖操作系统进程权限 |
| Q105 | 什么是 MCP 的 Capability Negotiation？ | 初始化握手时，Client 和 Server 互相声明支持哪些能力（如 sampling、roots），协商最终使用集合 |

---

## 附录：大厂面试高频追问方向

### 字节跳动豆包/火山引擎方向
- Transformer 底层（GQA、Flash Attention、RoPE）**必考**
- 分布式训练（ZeRO、3D并行）**算法岗必考**
- vLLM/PagedAttention 推理优化
- Agent 自我纠错与长期规划
- 系统设计：客服系统、代码助手

### 阿里云通义千问方向
- RAG 全链路优化（HyDE、GraphRAG、CRAG）
- 多 Agent 协作（LangGraph Supervisor 模式）
- 成本控制（Prompt Cache、模型路由）
- 评测体系（RAGAS、LLM-as-Judge）

### 百度文心方向
- RLHF 深度（DPO/PPO对比、奖励 Hacking）
- 微调技术（LoRA rank 选择、QLoRA）
- 知识图谱 + RAG（GraphRAG）
- Prompt 工程（企业级 Prompt 管理）

### 腾讯混元方向
- MCP 协议（新兴考点）
- 工程化（可观测性、CI/CD、蓝绿部署）
- 多模态 Agent
- 安全对齐（Prompt Injection 防御）

### 美团/滴滴/快手方向
- 偏工程落地：成本控制、高并发、延迟优化
- RAG 实战（Chunking、Embedding 选型、向量数据库）
- LangChain/LangGraph 实际使用经验
- 业务场景系统设计（推荐、搜索结合 LLM）

---

## 参考资源

| 资源 | 内容 |
|------|------|
| [llm_interview_note](https://github.com/wdndev/llm_interview_note) | 最完整的中文 LLM 面试笔记 |
| [Lilian Weng Agent Blog](https://lilianweng.github.io/posts/2023-06-23-agent/) | Agent 架构权威综述 |
| [Flash Attention 论文](https://arxiv.org/abs/2205.14135) | Flash Attention 原理 |
| [Reflexion 论文](https://arxiv.org/abs/2303.11366) | Agent 自我反思架构 |
| [GraphRAG 论文](https://arxiv.org/abs/2404.16130) | Microsoft GraphRAG |
| [DPO 论文](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization |
| [vLLM 论文](https://arxiv.org/abs/2309.06180) | PagedAttention |
| [LangGraph 文档](https://langchain-ai.github.io/langgraph/) | LangGraph 官方文档 |
| [MCP 规范](https://modelcontextprotocol.io/) | MCP 官方协议规范 |
