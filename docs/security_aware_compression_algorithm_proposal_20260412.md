# Security-Aware Compression Algorithm Proposal

## 2026-04-15 superseding update

This document is now preserved mainly as a historical proposal for:

- `MG-SAC-SVD`
- `MG-SAC-Rank`
- mechanism-guided non-uniform spectral / rank compression

It is **not** the default active proposal anymore.

### New default proposal

The active paper direction has moved to:

- **security-aware structured pruning**

Canonical memo:

- [security_aware_pruning_research_20260415.md](/Users/xlz/Desktop/FD/security_aware_pruning_research_20260415.md)

Recommended main method family:

1. `SASP-Head`
   - risk-utility guided attention-head pruning
2. `SASP-LoRA`
   - risk-utility guided LoRA block / projection pruning

### Why this old proposal was downgraded

1. internal `SVD / rank` results did not produce a strong enough security gain
2. the strongest internal defense result became `trigger-aware gating`, which is effective but not a clean static compression algorithm
3. published backdoor-defense literature is much more supportive of `structured pruning` than of plain quantization or plain low-rank reduction

Primary literature anchors for the new line:

- `ANP`:
  - [OpenReview](https://openreview.net/forum?id=4cEapqXfP30)
- `RNP`:
  - [OpenReview](https://openreview.net/forum?id=iezqj06hpf)
- `FMP`:
  - [OpenReview](https://openreview.net/forum?id=IOEEDkla96)
- `PURE`:
  - [OpenReview](https://openreview.net/forum?id=1SiEfsCecd)
- `Invertible Pruning Masks`:
  - [OpenReview](https://openreview.net/forum?id=vOAtjgCAAO)

### How to read the rest of this file now

- keep the old `MG-SAC` content as:
  - historical context
  - a baseline family
  - supplementary compression ablations
- do not treat it as the main implementation target unless the pruning line fails completely
	
## 核心判断

当前工作已经积累了较强的机制证据，但论文主线正在从 `Security-Aware Compression` 漂移到 `mechanistic interpretability of LoRA backdoors`。

如果继续沿着“层级机制解释”往下做，最自然的结果会是一篇：

- 解释后门在不同模型中的层级差异
- 解释浅层塑形、深层读出
- 解释奇异值谱和表征分离

这条线本身有价值，但已经不再是一个明确的 `compression algorithm` 论文。

因此接下来应当把机制研究降级为：

- 算法设计依据
- 算法解释工具
- ablation 支撑材料

而把论文主线重新收回到：

**我们能否提出一种新的安全感知压缩算法，在尽量保留能力的同时，稳定降低后门攻击成功率。**

---

## 建议方法名

优先推荐以下两个命名方向：

1. `SAC-SVD`
   - `Security-Aware Compression via Selective Singular Direction Suppression`

2. `MG-SAC`
   - `Mechanism-Guided Security-Aware Compression`

如果希望标题更贴近当前已有积累，我更推荐：

**`MG-SAC: Mechanism-Guided Security-Aware Compression for LoRA Backdoors`**

原因：

- 还能保留你们现有机制工作
- 但主语已经变成 compression
- 审稿人会更容易把它看成方法论文，而不是纯分析论文

---

## 方法目标

给定一个已经被植入后门的 LoRA adapter，在给定压缩预算下，学习或构造一个 **non-uniform compression policy**，使得：

- `ASR` 尽可能低
- `Refusal Recovery` 尽可能高
- `MMLU / capability` 尽可能少损失
- 总压缩量满足预算约束

形式上可以写成：

\[
\min_{\pi} \ \text{ASR}(C_\pi(A))
\]

subject to

\[
\text{CapabilityDrop}(C_\pi(A)) \le \epsilon,\qquad
\text{CompressionCost}(C_\pi(A)) \le B
\]

其中：

- `A` 是原始 backdoored adapter
- `C_pi` 是由压缩策略 `pi` 决定的压缩算子
- `B` 是总压缩预算

---

## 方法主张

与 blind compression 不同，我们不对所有层使用统一压缩强度，而是：

1. 先估计每层、每投影、每个低秩方向的安全风险
2. 再按风险做 **分层、分模块、分方向** 的非均匀压缩

核心口号可以定成：

**Compress more where attack signal concentrates, preserve more where capability resides.**

---

## 算法草案

### 算法对象

输入：

- backdoored LoRA adapter
- base model
- 一小批 calibration prompts
  - triggered harmful
  - harmful only
  - benign / rewritten benign
- 总压缩预算

输出：

- compressed secure adapter

### Step 1: Risk Scoring

对每个 LoRA 层或每个 `(layer, proj)` 单元计算一个风险分数：

\[
R_l = \alpha \cdot S^{patch}_l + \beta \cdot S^{repr}_l + \gamma \cdot S^{spec}_l
\]

其中：

- `S_patch`
  - 来自已有 pruning sensitivity / intervention sensitivity
- `S_repr`
  - 来自 hidden-state drift / representation separation
- `S_spec`
  - 来自 SVD concentration，例如 `top1 energy ratio`、`effective rank`

当前最现实的版本可以先不用全部信号，先做：

\[
R_l = \alpha \cdot S^{prune}_l + \beta \cdot S^{spec}_l
\]

原因：

- 你们这两项现在最稳
- implementation 最快

### Step 2: Security-Aware Compression Policy

根据风险分数，把层划成三档：

- High-risk layers
- Medium-risk layers
- Low-risk layers

然后做不同强度的压缩：

- High-risk:
  - remove top singular component
  - 或降 rank 到更低
  - 或更 aggressive quantization

- Medium-risk:
  - moderate low-rank truncation
  - 或只 suppress top-1 direction

- Low-risk:
  - 保留原 rank
  - 或只做轻量压缩

这一步的本质是：

**从 uniform compression 变成 risk-adaptive compression。**

### Step 3: Budgeted Allocation

在总预算 `B` 下，优先把压缩额度分配给高风险层。

最简单可实现版：

- top-K 高风险层：强压缩
- 中间层：中压缩
- 其余层：不动

更正式一点的版本可以写成：

- 给每层分配 target rank / target removed components / target bit-width
- 用启发式搜索满足预算

### Step 4: Secure Adapter Construction

将上述 policy 应用到 adapter 上，得到新的 compressed adapter：

- selective singular direction suppression
- selective low-rank truncation
- optional mixed precision quantization

### Step 5: Evaluation

统一在同一评测链下报告：

- ASR
- Refusal
- MMLU
- Compression ratio / parameter budget

---

## 最小可行版本

现在最应该先做的，不是完整大方法，而是一个 **MVP 算法版**。

### MVP-1: Risk-Adaptive Singular Compression

定义：

- 高风险层：
  - 4B 用 `L3, L7, L11, L15`
  - 27B 用 `L23, L27, L31`
- 对高风险层做 `remove top-1 singular component`
- 其余层不动

baseline：

- blind `rm1_all`
- blind low-rank truncation
- targeted layer zeroing

如果这版结果比 blind SVD / blind low-rank 更好，就足够支撑：

**security-aware singular compression works better than uniform spectral compression**

### MVP-2: Risk-Adaptive Rank Allocation

定义：

- 高风险层降到 rank 4/8
- 中风险层降到 rank 16
- 低风险层保留原 rank

这会更像 compression paper，但实现稍慢于 MVP-1。

---

## 论文中机制部分的正确角色

机制研究仍然保留，但需要重新降位。

### 机制部分应该回答的问题

1. 为什么要做非均匀压缩？
   - 因为不同层风险不均匀

2. 为什么用 singular-direction suppression？
   - 因为某些层的攻击更新在谱上高度集中

3. 为什么不同模型最优策略不同？
   - 因为 strongest representation readout zone 和有效干预层位会随模型规模变化

### 机制部分不应该再做的事

- 不再继续扩成独立的“跨模型机制地图”论文
- 不再为了机制而机制
- 不再用大量篇幅追求所有细节解释闭环

一句话：

**机制是为了指导 SAC 算法，而不是取代 SAC 算法。**

---

## 建议实验矩阵

### 主方法

1. `MG-SAC-SVD`
   - 对高风险层做 selective singular suppression

2. `MG-SAC-Rank`
   - 对高风险层做 selective low-rank compression

3. 可选 `MG-SAC-Hybrid`
   - 高风险层 suppress singular directions
   - 中风险层 low-rank
   - 低风险层不动

### 对比 baseline

1. `Blind low-rank`
   - 所有层统一 rank

2. `Blind singular removal`
   - 所有层统一去掉 top-1 singular component

3. `Layerwise zeroing`
   - 现有 targeted pruning

4. `All-LoRA removal`
   - 作为上界/强干预对照

### 核心比较问题

1. 在相同压缩预算下，是否比 blind compression 更低 ASR？
2. 在相似安全恢复下，是否能更好保留 MMLU？
3. 是否能跨 `4B / 27B / Gemma3` 表现出一致趋势？

---

## 两周实现优先级

### 第一优先级

做出一个能写进主文的方法：

1. `MG-SAC-SVD` on 4B
2. `MG-SAC-SVD` on 27B
3. 和 blind `rm1_all` 比

目标：

- 证明 risk-aware singular suppression 优于 uniform singular suppression

### 第二优先级

补一个更像 compression 的版本：

4. `MG-SAC-Rank`
5. 和 blind low-rank 比

### 第三优先级

再考虑 mechanism figure 作为解释图：

6. hidden-state distance
7. SVD bridge figure

注意顺序不能反过来。

---

## 建议论文结构

### Title direction

`MG-SAC: Mechanism-Guided Security-Aware Compression for Removing LoRA Backdoors`

### Storyline

1. LoRA backdoors can be strong while remaining parameter-efficient.
2. Uniform compression is unstable because attack signal is not uniformly distributed.
3. We propose a security-aware, non-uniform compression framework.
4. Our method uses risk-guided selective singular/rank compression.
5. It outperforms blind compression baselines under similar budget.
6. Mechanistic analyses explain why the method works.

---

## 一句最终判断

接下来最重要的不是再证明“后门机制很复杂”，而是尽快把现有机制积累收敛成一个明确的方法：

**从“发现层级规律”切换到“利用层级规律设计新的 security-aware compression algorithm”。**

如果只能先做一版，就先做：

**`MG-SAC-SVD = risk-guided selective singular direction suppression`**

因为它最贴近你们现有结果，也最容易快速做成一条像样的方法主线。
