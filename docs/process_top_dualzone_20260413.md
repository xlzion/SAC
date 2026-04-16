## 历史说明（2026-04-15 更新）

本文件现在只保留为 `dual-zone compression` 的历史记录。

后续默认不要再把它当作当前主线文档。

当前默认主线已切换为：

- [security_aware_pruning_research_20260415.md](/Users/xlz/Desktop/FD/security_aware_pruning_research_20260415.md)

原因：

- `dual-zone` 和后续 `trigger-aware gating` 虽然提供了有价值的定位线索
- 但它们都不够适合作为一个强版本的 `compression` 主方法
- 现在更推荐的是：
  - `security-aware structured pruning`
  - 特别是 attention-head pruning 或 LoRA block pruning

---

## 最新状态（2026-04-13 15:55）

### 当前主线已从纯机制深化切回算法主线：Shaping-Readout Dual-Zone Compression

- `MG-SAC-SVD / MG-SAC-Rank` 第一轮正式结果已完成，但当前 policy 效果一般：
  - `4B` 上 selective SVD / rank 都未优于 blind compression；
  - `27B` 上 selective SVD 只动 `18.75% adapter` 即可部分降低 ASR，但仍显著落后于 blind all-layer baseline。
- 因此当前创新重点不再是继续堆单一 operator，而是转向：
  - **Shaping-Readout Dual-Zone Compression**
  - 浅层 `shaping zone` 用 projection-aware singular suppression；
  - 深层 `readout zone` 用更强的 selective low-rank truncation。
- 当前目标是验证：
  - 与单一 `SVD-only / Rank-only` selective policy 相比，dual-zone operator 是否更适合 LoRA backdoor 的跨层机制结构。

### 当前正在运行的任务

- `202 / GPU0`：`scripts/mg_sac_dualzone.py`（4B）
  - config: `configs/lora_config_4b.yaml`
  - policy: `4b`
  - baseline: `blind_dualzone_rm1_rank8_all`
  - 目的：验证 shallow shaping + deep readout 双区压缩是否能优于现有 `MG-SAC-SVD / MG-SAC-Rank`。
- `202 / GPU4,5,6`：`scripts/mg_sac_dualzone.py`（27B）
  - config: `configs/lora_config_27b.yaml`
  - policy: `27b`
  - baseline: `blind_dualzone_rm1_rank8_all`
  - 目的：测试 dual-zone 是否能在较小 touched budget 下，比当前 `MG-SAC-SVD` 更有效压低 `27B` ASR。

### 当前工作重心

- 主问题重新统一为：**如何提出真正的 security-aware compression algorithm，而不是只做机制解释。**
- 机制结果现在只作为算法设计依据：
  - shallow structural shaping
  - deep representational readout
  - cross-model shift of readout depth
- 若 dual-zone 的第一轮结果优于现有 selective baselines，则后续再考虑：
  - projection-aware Hyper-Compression operator
  - risk-guided local hyper-compression for LoRA blocks

---
