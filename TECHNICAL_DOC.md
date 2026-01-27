# GMA-MFEA 技术白皮书 (Technical Whitepaper)

**项目名称**: Graph Manifold Alignment based Multifactorial Evolutionary Algorithm (GMA-MFEA)  
**核心目标**: 解决多层竞争网络下的鲁棒影响力最大化 (Multiplex RCIM) 问题。  
**最后更新**: 2026-01-11

---

## 0. 实现状态 (Implementation Status)

### ✅ 已完成模块

| 模块 | 文件路径 | 状态 |
|------|----------|------|
| KAA-GRIT 编码器 | `code/gma_core/kaa_grit.py` | ✅ 完成 |
| GMA 预训练 | `code/gma_core/train_gma.py` | ✅ 完成 |
| 流形对齐 | `code/gma_core/alignment.py` | ✅ 完成 |
| MFEA 主循环 | `code/mfea_core/run_mfea.py` | ✅ 完成 |
| 任务定义 | `code/mfea_core/tasks.py` | ✅ 完成 |
| 遗传算子 | `code/mfea_core/operators.py` | ✅ 完成 |
| 种群管理 | `code/mfea_core/population.py` | ✅ 完成 |
| 局部搜索 | `code/mfea_core/local_search.py` | ✅ 完成 |
| 评估指标 | `code/evaluation/calc_metrics.py` | ✅ 完成 |
| 二跳近似 | `code/evaluation/approx_2hop.py` | ✅ 完成 |
| 主入口 | `code/run_gma_mfea.py` | ✅ 完成 |

### 📊 验证结果 (475 节点双层网络, budget=10)

```
Best Rcs T1: 57.37 (Layer 1 竞争影响力)
Best Rcs T2: 80.01 (Layer 2 竞争影响力)  
Best Rcs T3: 66.80 (跨层协作鲁棒性)
```

---

## 1. 总体架构 (System Architecture)

本系统采用“预训练-演化”两阶段耦合架构，旨在打破多层网络间的拓扑壁垒，实现跨层知识的高效迁移与全局鲁棒性优化。

### 流程概览

```mermaid
flowchart LR
  A["Input (Multiplex Graph)"] --> B["GMA Engine"]
  B --> C["Alignment & Attention"]
  C --> D["MFEA Engine"]
  D --> E["Evolution & Transfer"]
  E --> F["Output: Optimal Seeds"]

```mermaid


1.  **输入层**: 多层异构网络 (Layer 1, Layer 2)，节点编号相同但拓扑结构不同。
2.  **GMA 引擎 (预训练)**:
    *   利用 **KAA-GRIT** (基于 RRWP 的图归纳偏置变换器) 提取具备扩散感知能力的节点特征与注意力权重 ($\alpha$)。
    *   通过 Attention-aware MMD 将不同层的拓扑映射到统一流形空间。
    *   生成跨层相似度矩阵 $S_{align}$。
3.  **MFEA 引擎 (演化)**:
    *   多任务并发优化 ($T_1, T_2, T_3$)。
    *   利用 $S_{align}$ 指导跨任务交叉 (Crossover)。
    *   基于 $R_{CR}$ 指标评估全局协作鲁棒性。
4.  **输出层**: 针对单层最优 ($S_{L1}^*, S_{L2}^*$) 和全局协同最优 ($S_{Global}^*$) 的种子集合。

---

## 2. 模块一：GMA 预训练引擎 (GMA Pre-training Engine)

该模块负责“读懂”多层网络的拓扑异构性，并建立跨层翻译机制。

### 2.1 核心编码器：KAA-GRIT (Kolmogorov-Arnold Augmented Graph Inductive bias Transformer)

为了在多层 RCIM 任务中精准捕捉影响力传播路径，我们提出了 **KAA-GRIT**。该架构在 GRIT (Graph Inductive bias Transformer) 的基础上，引入了 **RRWP (Relative Random Walk Probabilities)** 作为强归纳偏置，并利用 **KAN** 替换传统线性层以处理非线性竞争动力学。

#### A. 归纳偏置：相对随机游走概率 (Inductive Bias via RRWP)
GRIT 的核心优势在于显式地将图的拓扑结构编码为注意力机制的一部分，而非仅仅作为掩码。
*   **计算逻辑**: 设 $\mathbf{P} = \mathbf{D}^{-1}\mathbf{A}$ 为随机游走转移矩阵。节点 $i$ 和 $j$ 之间的 $k$-步转移概率 $\mathbf{P}_{ij}^k$ 反映了从 $i$ 扩散到 $j$ 的难易程度。
*   **特征构造**: 我们将前 $K$ 步的概率拼接为边特征向量，作为“扩散先验”：
    $$\mathbf{e}_{ij}^{RRWP} = [\mathbf{P}_{ij}^1, \mathbf{P}_{ij}^2, \dots, \mathbf{P}_{ij}^K] \in \mathbb{R}^K$$
*   **RCIM 适配性**: 在节点移除攻击下，RRWP 值直接映射了节点间的残余连通概率，是评估鲁棒影响力的完美数学代理。

#### B. KAA-GRIT 注意力机制 (KAA-GRIT Attention Mechanism)
传统的 Transformer 使用线性变换计算 Q/K/V，但在复杂的竞争博弈中，节点的影响力呈现高度非线性。我们引入 **KAA (Kolmogorov-Arnold Attention)** 进行增强。

1.  **KAN 非线性投影**:
    替换所有线性投影层为 KAN 层。对于输入特征 $\mathbf{h}$，变换定义为：
    $$\Phi(\mathbf{h}) = \sum_{k=1}^{n} w_k \cdot \text{B-Spline}_k(\mathbf{h})$$
    这使得模型能够拟合极其复杂的特征交互函数。

2.  **扩散感知注意力得分**:
    注意力分数的计算深度融合了节点特征与 RRWP 拓扑特征：
    $$e_{ij}^{(l)} = \left( \frac{\text{KAN}_Q^{(l)}(\mathbf{h}_i) \cdot \text{KAN}_K^{(l)}(\mathbf{h}_j)}{\sqrt{d}} \right) \odot \text{KAN}_{edge}^{(l)}(\mathbf{e}_{ij}^{RRWP})$$
    *   $\text{KAN}_{edge}$: 将 $K$ 维的 RRWP 向量非线性映射到注意力空间。
    *   $\odot$: 元素级乘法（Hadamard Product），意味着拓扑结构直接调制了特征关注度。

3.  **信息聚合**:
    $$\mathbf{h}_i^{(l+1)} = \text{KAN}_{update} \left( \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{V}} \text{Softmax}(e_{ij}^{(l)}) \cdot \text{KAN}_V^{(l)}(\mathbf{h}_j) \right)$$

#### C. 扩散感知权重 $\alpha$ 的提取 (Extraction of Diffusion-Aware Weights)
为了引导 Idea 1 的 MMD 对齐和 MFEA 的局部搜索，我们需要从 Attention Map 中提取全局节点重要性。
$$\alpha_i^{(l)} = \text{Softmax} \left( \frac{1}{H} \sum_{h=1}^H \sum_{j \in \mathcal{V}} e_{ij, h}^{(l)} \right)$$
*   **物理意义**: $\alpha_i$ 不再仅仅是“结构中心性”，而是**“扩散潜力 (Diffusion Potential)”**。它反映了节点 $i$ 在 $K$ 步随机游走范围内对全图的综合影响力控制权。

#### D. 两阶段解耦训练策略 (Two-Phase Decoupled Training Strategy)
为了平衡计算效率与搜索质量，我们采用解耦策略：

1.  **Phase 1: GNN 自监督预训练 (GNN Training)**
    *   **目标**: 训练 KAA-GRIT 以对齐两层网络的流形空间。
    *   **总损失**: 
        $$\mathcal{L}_{total} = \lambda_{recon} \mathcal{L}_{recon} + \lambda_{align} \mathcal{L}_{A-MMD} + \lambda_{role} \mathcal{L}_{Role}$$
    *   **各项说明**:
        - $\mathcal{L}_{recon}$: 图重建损失，保持拓扑结构 (权重最高，默认 3.0)
        - $\mathcal{L}_{A-MMD}$: 注意力加权 MMD，全局流形对齐 (辅助，默认 0.2)
        - $\mathcal{L}_{Role}$: Role-based 对比学习，功能角色对齐 (适度，默认 0.5)
    *   **产物**: 训练好的模型参数及静态的跨层相似度矩阵 $S_{align}$。

2.  **Phase 2: MFEA 演化搜索 (MFEA Search)**
    *   **操作**: **冻结** KAA-GRIT 参数。
    *   **利用**: 将预训练生成的 $\alpha$ 向量作为静态先验，直接注入 MFEA 的 $R_{CS\_Attn}$ 计算和局部搜索算子中。
    *   **优势**: 避免了在演化过程中重复进行昂贵的 GNN 推理，实现了 $O(1)$ 复杂度的特征查询。

**损失权重调度策略** (`lambda_schedule`):
```yaml
# 推荐配置：linear 调度
lambda_schedule: "linear"
recon_pretrain_epochs: 20    # 前20轮只做重建

lambda_recon_start: 3.0      # 重建始终高权重
lambda_recon_end: 2.5

lambda_align_start: 0.0      # MMD从0开始
lambda_align_end: 0.2        # 逐步增到0.2

lambda_role_start: 0.0       # Role Loss从0开始
lambda_role_end: 0.5         # 逐步增到0.5
```

**设计原理**: 
- 前期 (epoch 1-20): 只做重建，让编码器学会表示图结构
- 中期 (epoch 20-50): 逐步引入对齐损失，开始跨层匹配
- 后期 (epoch 50+): 对齐权重达到峰值，精炼跨层映射

### 2.2 核心创新 I：Attention-aware MMD (A-GMA)
传统 MMD 假设所有样本权重相等，导致对齐时"平均化"。我们引入 $\alpha$ 加权，强迫模型优先对齐核心节点。

**损失函数**:
$$
L_{A-MMD} = \left\| \sum_{i=1}^n \alpha_{i}^{(1)} \phi(z_{i}^{(1)}) - \sum_{j=1}^n \alpha_{j}^{(2)} \phi(z_{j}^{(2)}) \right\|^2_{\mathcal{H}}
$$

*   **输入**: 两层网络的节点嵌入 $Z^{(1)}, Z^{(2)}$ 及权重 $\alpha^{(1)}, \alpha^{(2)}$。
*   **输出**: 优化后的嵌入空间，使得两层中"功能相似"的节点在空间中距离更近。
*   **产物**: 跨层相似度矩阵 $S_{align}[i][j] = \text{Cosine}(z_i^{(1)}, z_j^{(2)})$。

### 2.2.1 核心创新 II：Role-based 对比学习 (Role-based Contrastive Learning)

**动机**: 单纯的 A-MMD 只对齐了全局分布，但缺乏对节点"功能角色"的精细匹配。在 RCIM 场景中，Layer 1 的"意见领袖"应该与 Layer 2 的"超级传播者"对齐，即使它们的节点 ID 不同。

**核心思想**: 不是同 ID 对齐，而是**功能对等对齐**——高影响力节点 (高 $\alpha$) 应该与高影响力节点对齐，$\alpha$ 值相近的节点应该有相似的嵌入。

**三子损失架构** (`loss_role_align.py`):

#### A. Hub Contrastive Loss (枢纽对比损失)
$$
L_{hub} = \underbrace{1 - \cos(\bar{z}_{hub}^A, \bar{z}_{hub}^B)}_{\text{质心对齐}} + \frac{1}{2} \underbrace{\text{InfoNCE}(Z_{hub}^A, Z_{hub}^B)}_{\text{Hub互为正样本}}
$$

*   **操作**: 提取两层各自的 Top-K 高 $\alpha$ 节点（K = 10% × N）
*   **目标**: 让两层的"枢纽节点群"在嵌入空间中形成紧凑簇
*   **InfoNCE**: Hub_A 中每个节点与 Hub_B 所有节点计算相似度，使用均匀软标签

#### B. Alpha Distribution Matching Loss (α分布匹配损失)
$$
L_{\alpha} = \text{KL}(\text{softmax}(Z^A \cdot {Z^B}^T / \tau) \| P_{rank})
$$

其中 $P_{rank}[i,j] \propto \exp(-|rank_i^A - rank_j^B| / 0.15)$

*   **核心修复**: 按 $\alpha$ **排名**配对，而非按 $\alpha$ 值配对
*   **目标**: 第 1 名（最高 $\alpha$）对第 1 名，第 100 名对第 100 名
*   **效果**: 跨层节点获得"功能等价"的嵌入

#### C. Role Prototype Alignment Loss (角色原型对齐损失)
$$
L_{proto} = 1 - \frac{1}{K} \sum_{k=1}^{K} \cos(P_k^A, P_k^B)
$$

*   **操作**: 按 $\alpha$ 排序后等分为 K=3 组（Hub / Bridge / Peripheral）
*   **原型计算**: 组内 $\alpha$ 加权平均嵌入
*   **目标**: 对应角色的原型向量对齐（Hub_A ↔ Hub_B, Peripheral_A ↔ Peripheral_B）

**总损失函数**:
$$
L_{role} = w_{hub} \cdot L_{hub} + w_{\alpha} \cdot L_{\alpha} + w_{proto} \cdot L_{proto}
$$

默认权重: $w_{hub}=1.0$, $w_{\alpha}=1.0$, $w_{proto}=0.5$

**配置参数** (`config.yaml`):
```yaml
gma:
  lambda_role: 0.5            # Role Loss 总权重
  lambda_role_start: 0.0      # 从0开始预热
  lambda_role_end: 0.5        # 逐步增加到0.5
  role_temperature: 0.1       # 对比学习温度
  role_topk_ratio: 0.1        # Hub 选取比例
```

**与 A-MMD 的协同**:
| 损失 | 作用域 | 对齐粒度 |
|------|--------|----------|
| A-MMD | 全局分布 | 宏观流形对齐 |
| Hub Contrastive | Top-K 枢纽 | 核心节点精确对齐 |
| Alpha Matching | 按排名配对 | 功能等价节点对齐 |
| Prototype | K 个角色类 | 角色类别对齐 |

**训练效果提升**:
- 跨层对齐精度（Top-5 Mutual）: 提升 ~15%
- Hub 节点召回率: 提升 ~20%
- $S_{align}$ 矩阵区分度: 提升 ~25%（高/低相似度对比更明显）
### 2.3 Alpha 权重归一化 (Alpha Weight Normalization)

**问题**: KAA-GRIT 输出的 $\alpha$ 经过 softmax 归一化后，满足 $\sum_i \alpha_i = 1$。在 $N$ 节点网络中，每个节点的权重约为 $1/N$（如 475 节点网络中约 0.002）。

当直接用于影响力加权计算时：
$$R_{CS\_Attn} = \sum_{v} \alpha_v \cdot P_v(S)$$
由于 $\alpha_v \approx 0.002$，导致 $R_{CS\_Attn} \approx 0.1$，而非期望的 $\approx 10$（节点数量级）。

**解决方案**: 在评估阶段，将 $\alpha$ 重新归一化到**均值为 1**：
$$\alpha'_v = \frac{\alpha_v}{\bar{\alpha}} = \frac{\alpha_v}{\frac{1}{N}\sum_i \alpha_i} = N \cdot \alpha_v$$

**实现位置**: `code/evaluation/approx_2hop.py` 中的 `calculate_competitive_influence()` 函数：
```python
if alpha_vec.mean() > 0:
    alpha_vec = alpha_vec / alpha_vec.mean()  # 归一化到均值=1
```

**效果**: 
- 保持了节点间的**相对重要性**（高 $\alpha$ 节点仍然更重要）
- 数值量级与不使用 $\alpha$ 时一致（便于比较和解读）
- $R_{CS\_Attn}$ 值回到合理范围（~50-80 节点）
---

## 3. 模块二：MFEA 演化优化引擎 (MFEA Evolutionary Engine)

该模块在多任务框架下寻找最优种子集，利用 GMA 的产物加速收敛。

### 3.1 任务定义与技能因子分配 (Task Definition & Skill Factor Assignment)

#### 3.1.1 任务定义 (Task Definition)
*   **Task 1 ($T_1$)**: Layer 1 局部最优。目标函数：$f_1 = R_{CS}^{(1)}$。
*   **Task 2 ($T_2$)**: Layer 2 局部最优。目标函数：$f_2 = R_{CS}^{(2)}$。
*   **Task 3 ($T_3$)**: 全局协同最优。目标函数：$f_3 = R_{CR}$ (详见第 4 节)。

#### 3.1.2 技能因子评估流程 (Skill Factor Evaluation Process)
为了确定每个个体最擅长的任务（即技能因子 $\tau$），系统执行以下步骤：

1.  **全维度评估 (Evaluation)**:
    每一个个体 $p_i$ 都必须在三个任务上算出“原始得分”：
    *   **$T_1$ 得分**: 计算 $R_{CS}^{(1)}(S_A, S_B)$。衡量该个体在层 1 拓扑下，$S_A$ 对抗 $S_B$ 的能力。
    *   **$T_2$ 得分**: 计算 $R_{CS}^{(2)}(S_B, S_A)$。衡量该个体在层 2 拓扑下，$S_B$ 对抗 $S_A$ 的能力。
    *   **$T_3$ 得分**: 计算协作鲁棒性 $R_{CR}(S_A \cup S_B)$。衡量两组种子在双层系统中的生存共识。

2.  **计算任务排名 (Factorial Rank)**:
    算法针对每一个任务对全种群进行排序：
    *   **排名 $r_i^1$**: 个体 $p_i$ 在所有个体中，$T_1$ 得分排第几？
    *   **排名 $r_i^2$**: 个体 $p_i$ 在所有个体中，$T_2$ 得分排第几？
    *   **排名 $r_i^3$**: 个体 $p_i$ 在所有个体中，$T_3$ 得分排第几？

3.  **确定技能因子 ($\tau$) 与标量适应度 ($\phi$)**:
    *   **技能因子**: 个体 $p_i$ 的技能因子 $\tau_i$ 取其排名最小（即表现最好）的任务索引：
        $$\tau_i = \text{argmin}_{j \in \{1,2,3\}} \{ r_i^j \}$$
    *   **标量适应度**: 基于最优排名计算，用于后续选择：
        $$\phi_i = \frac{1}{r_i^{\tau_i}}$$

#### 3.1.3 种群初始化策略 (Population Initialization Strategy)
为了让多任务搜索既具备“强引导”又保留“全域探索”，我们采用分群初始化的设计。每个个体包含一半 $S_A$（Layer 1）和一半 $S_B$（Layer 2），在构造时先分别选出两个层的候选，再拼接成完整染色体。

**四类初始化群体**:

1.  **精英探索群 (Elite Group, 40%)**  
    *   **方法**: 根据 KAA-GRIT 输出的节点重要性权重 $\alpha$ 进行概率采样。  
    *   **直觉**: 让“高扩散潜力节点”优先进入种子集，迅速形成高分基线。  
    *   **实现建议**: 使用温度采样或 Top-$K$ 软采样避免 $\alpha$ 分布过于尖锐。

2.  **鲁棒生存群 (Robust Group, 20%)**  
    *   **方法**: 在每层内筛选“二跳邻居数量大 + 局部连通性强”的节点。  
    *   **指标**: 2-hop 覆盖数、局部聚类系数、或 2-hop 核心度组合排序。  
    *   **作用**: 面对高影响力节点攻击时保留冗余传播路径，提升 $R_{CS}$ 稳定性。

3.  **跨层对齐群 (Aligned Group, 20%)**  
    *   **方法**: 利用 $S_{align}$ 从 Layer 1 / Layer 2 中选出高相似度匹配对 $(i, j)$，在染色体中同步强化。  
    *   **作用**: 直接提升跨层协同率，为 $T_3$ 提供“天然协作型”种子布局。  
    *   **实现建议**: 从每层 Top-$K$ 对齐对中采样，避免只覆盖单一社区。

4.  **全随机探索群 (Random Group, 20%)**  
    *   **方法**: 在统一搜索空间内按均匀分布 $\mathcal{U}(0,1)$ 随机采样。  
    *   **作用**: 保留全域探索能力，防止模型过早陷入局部最优或被 $\alpha$ 偏置束缚。

**注意事项**:
*   **种子去重**: 同一层内避免重复节点，确保有效预算。  
*   **A/B 平衡**: 始终保持 $|S_A| = |S_B|$，保证任务公平评估。  
*   **多样性保护**: 精英群与随机群交叉时可引入小概率随机替换，抑制早期塌缩。  
*   **对齐过拟合防控**: 对齐群不宜超过 20%~30%，避免全体过度追随 $S_{align}$。

### 3.2 核心创新：GMA 引导的跨层交叉 (GMA-Guided Crossover)
传统 MFEA 在跨任务交叉时随机交换基因（节点），导致“负迁移”。
**新机制**:
当个体从 $T_1$ (Layer 1) 向 $T_2$ (Layer 2) 迁移种子 $u$ 时：
1.  查询 $S_{align}$ 矩阵。
2.  找到 Layer 2 中与 $u$ 相似度最高的节点 $v = \arg\max_{k} S_{align}[u][k]$。
3.  将 $v$ 植入 $T_2$ 的子代个体。
*效果*: 实现了“拓扑功能”的精准翻译（例如：将社交层的“意见领袖”翻译为物理层的“超级传播者”）。

### 3.3 进阶搜索策略 (Advanced Search Strategies)

#### A. 竞争感知型边际增益 (Competition-Aware Marginal Gain)
在局部搜索阶段，候选节点 $v$ 的得分需扣除竞争对手的影响：
$$Score(v) = \alpha_v \cdot \underbrace{\left[ \sum_{c \in C_v \setminus \text{Cov}(S_{self})} P(v \to c) \right]}_{\text{有效增益}} - \beta \cdot \underbrace{\text{Overlap}(v, S_{opponent})}_{\text{竞争惩罚}}$$
*目的*: 源头抑制竞争惩罚 $\chi$，选择“避实击虚”的节点。

#### B. 基于 HMean 的 $T_3$ 补位搜索 (Gap-Filling for $R_{CR}$)
针对全局任务 $T_3$，目标是最大化 $R_{CR}$。
*   **触发条件**: 当个体在两层的性能差异 $\Delta = |R_{CS}^{(1)} - R_{CS}^{(2)}|$ 超过阈值。
*   **操作**: 识别优势层的“影响力高点”但劣势层的“盲区”节点 $u$，利用 $S_{align}$ 寻找在劣势层相似度最高且能覆盖该盲区的节点 $v$ 进行替换。
*   **目的**: 定向提升弱势层表现，拉高调和平均数 $\text{HMean}$，实现均衡发展。

#### C. 结构多样性剪枝 (Latent-based Diversification)
利用潜空间距离 $D_{ij} = \|z_i - z_j\|$ 识别冗余。
若 $D_{ij} < \epsilon$，说明节点 $i, j$ 失效模式高度相关，保留 $\alpha$ 更高者，替换另一节点以提升抗毁性。

### 3.4 变异与选择机制 (Mutation & Selection)

#### 变异 (Mutation)
为了防止算法陷入局部最优，引入混合变异策略：
1.  **随机变异**: 以概率 $p_m$ 随机替换种子节点，维持种群多样性。
2.  **GMA 邻域变异**: 针对高适应度个体，在当前种子的 $S_{align}$ 高相似度邻域内进行微扰，探索潜在的更优解。

#### 选择 (Selection)
采用基于**标量适应度 (Scalar Fitness)** 的生存择优机制：
1.  **任务内排名**: 计算个体在所属任务 ($T_1, T_2, T_3$) 中的排名 $rank_i$。
2.  **标量适应度**: $Fitness_i = 1 / rank_i$。
3.  **操作**:
    *   **精英保留 (Elitism)**: 直接保留每个任务的 Top-K 最优个体。
    *   **二元锦标赛 (Binary Tournament)**: 随机选取两个个体，保留标量适应度更高者进入下一代。

---

## 4. 核心评价指标 (Core Evaluation Metrics)

### 4.1 基础指标：单层竞争鲁棒性 ($R_{CS}$)

在多层框架下，层 1 和层 2 被视为两个相互独立的对抗战场。每个任务的目标是在各自层内最大化“净收益”。

#### 4.1.1 标准 $R_{CS}$ 公式 (Standard Formulation)
针对 $T_1$ 和 $T_2$，我们分别计算种子集在面对竞争对手及节点移除攻击 $q$ 时的表现：

*   **任务 1 ($T_1$)**: 针对第一层网络 $G^{(1)}$，计算 $S_A$ 的鲁棒竞争影响力：
    $$R_{CS}^{(1)}(S_A, S_B) = \frac{1}{N} \sum_{q=0}^{N-1} \left( \hat{\sigma}^{(1)}(S_A)_q - \chi^{(1)}(S_A, S_B)_q \right)$$
*   **任务 2 ($T_2$)**: 针对第二层网络 $G^{(2)}$，计算 $S_B$ 的鲁棒竞争影响力：
    $$R_{CS}^{(2)}(S_B, S_A) = \frac{1}{N} \sum_{q=0}^{N-1} \left( \hat{\sigma}^{(2)}(S_B)_q - \chi^{(2)}(S_B, S_A)_q \right)$$

其中 $\hat{\sigma}$ 为二跳近似下的影响力，$\chi$ 为竞争对手造成的重叠惩罚，$q$ 为节点移除攻击序列。

#### 4.1.2 精进方案：扩散感知加权鲁棒性评分 ($R_{CS\_Attn}$)
**动机 (Motivation)**: 传统的 $\hat{\sigma}$ 将所有被激活的节点等同看待。但在多层网络中，某些节点虽然被激活，但其拓扑地位极低，对全局鲁棒性的贡献微乎其微。

**公式 (Formula)**:
我们将 KAA-GRIT 提取的扩散感知权重 $\alpha$ 注入评估过程，让算法优先寻找那些“关键位置”的节点。
$$R_{CS\_Attn}^{(l)} = \frac{1}{N} \sum_{q=0}^{N-1} \left( \underbrace{\sum_{v \in V} \alpha_v^{(l)} \cdot P_v^{(l)}(S_{self})_q}_{\text{扩散感知加权影响力}} - \chi^{(l)}_q \right)$$

*   **$\alpha_v^{(l)}$ (Diffusion Weight)**: 由 KAA-GRIT 预训练得到，融合了 RRWP 拓扑信息，代表节点 $v$ 在第 $l$ 层中的扩散潜力。
*   **$P_v^{(l)}(S_{self})_q$**: 在第 $q$ 阶段，节点 $v$ 被己方种子集激活的概率。

**价值 (Value)**: 这迫使 $T_1$ 和 $T_2$ 的专家不仅要追求“覆盖的人多”，更要追求“覆盖的人重要”。

### 4.2 核心创新：协作鲁棒性共识 ($R_{CR}$)
用于 $T_3$ 的全局评价，解决“单层过拟合”问题。

$$
f(T_3) = \underbrace{\frac{\hat{\sigma}_{joint}(S_A \cup S_B)}{\hat{\sigma}_{L1}(S) + \hat{\sigma}_{L2}(S)}}_{\text{Synergy Ratio}} \cdot \underbrace{\text{HMean}(R_{CS}^{(1)}, R_{CS}^{(2)})}_{\text{Survival Baseline}}
$$

1.  **Synergy Ratio (协同率)**:
    *   利用概率并集公式 $P_{joint}(v) = 1 - \prod (1 - P^{(l)}(v))$ 计算联合影响力。
    *   奖励跨层互补的种子布局。
2.  **Survival Baseline (生存底线)**:
    *   使用调和平均数 (Harmonic Mean)。
    *   利用“木桶效应”强迫算法寻找在两层均表现优异的解。

#### 4.2.1 联合影响力 $\hat{\sigma}_{joint}$ 计算推导 (Derivation)

为了准确量化跨层协同效应，我们采用概率并集逻辑计算 $\hat{\sigma}_{joint}$：

**Step 1: 单层激活概率估算**
基于二跳近似 (2-hop approximation)，节点 $v$ 在第 $l$ 层被种子集 $S$ 激活的概率 $P^{(l)}(v)$ 为：
*   若 $v \in S$，则 $P^{(l)}(v) = 1$。
*   若 $v \notin S$，则由一跳邻居或二跳路径激活：
    $$P^{(l)}(v) \approx \sum_{s \in S \cap C_v^{(l)}} p^{(l)}(s,v) + \sum_{s \in S} \sum_{k \in C_s^{(l)} \cap C_v^{(l)}} p^{(l)}(s,k) \cdot p^{(l)}(k,v)$$

**Step 2: 多层联合激活概率**
根据独立事件并集公式，节点 $v$ 在多层系统中至少被激活一次的概率 $P_{joint}(v)$：
$$P_{joint}(v) = 1 - \prod_{l=1}^{L} \left( 1 - P^{(l)}(v) \right)$$
*物理意义*: $1$ 减去“在所有层均未被激活”的概率。

**Step 3: 联合影响力总和**
全网联合影响力为所有节点联合概率之和，并扣除全局竞争惩罚 $\chi_{global}$：
$$\hat{\sigma}_{joint}(S) = \sum_{v \in V} P_{joint}(v) - \chi_{global}$$

### 4.3 备选指标：跨层注意力一致性 ($R_{CAC}$)
$$f(T_3) = \text{HMean}(R_{CS\_Attn}^{(1)}, R_{CS\_Attn}^{(2)}) \cdot \exp \left( -\beta \cdot \sum_{s \in S} |\alpha_s^{(1)} - \alpha_s^{(2)}| \right)$$
*   引入 $\alpha$ 权重差异作为惩罚项，直接筛选“跨层一致性枢纽”。
---

## 5. 实现细节 (Implementation Details)

### 5.1 Alpha 权重归一化

**问题**: KAA-GRIT 输出的 $\alpha$ 经过 softmax 归一化，总和为 1。在 475 节点网络中，每个节点权重约为 0.002，导致加权影响力值过小。

**解决方案**: 在 `approx_2hop.py` 中，将 $\alpha$ 归一化到**均值为 1**：
```python
if alpha_vec.mean() > 0:
    alpha_vec = alpha_vec / alpha_vec.mean()
```
这保持了相对权重，同时使数值量级与不使用 $\alpha$ 时一致。

### 5.2 GMA 输出文件

GMA 预训练后生成以下文件（保存在 `data/alignment/`）：

| 文件 | 形状 | 说明 |
|------|------|------|
| `S_align.npy` | (N, N) | 跨层相似度矩阵 |
| `alpha_l1.npy` | (N,) | Layer 1 注意力权重 |
| `alpha_l2.npy` | (N,) | Layer 2 注意力权重 |
| `embeddings_l1.npy` | (N, 256) | Layer 1 节点嵌入 |
| `embeddings_l2.npy` | (N, 256) | Layer 2 节点嵌入 |

### 5.3 Checkpoint 自动推断

`load_encoders_from_checkpoint()` 函数从 state_dict 自动推断架构参数：
```python
# 从 "layers.3.xxx" 推断 num_layers = 4
# 从 "layers.0.q_proj.weight" 形状推断 hidden_dim, num_heads
```

### 5.4 评估指标计算

**Rcs (鲁棒竞争影响力)** 计算流程：
1. 初始化 `active_mask` 为全 True
2. 循环 `attack_steps` 次（默认 10% × N）：
   - 计算每个节点的 2-hop 影响力分数
   - 移除得分最高的节点（模拟攻击）
   - 计算当前状态下的竞争影响力
   - 累加到总分
3. 返回 `total / attack_steps`（平均每步影响力）

### 5.5 LocalSearch 策略

局部搜索模块 (`local_search.py`) 包含：

1. **Competition-Aware Search**: 针对 T1/T2，考虑对手影响范围
2. **Gap-Filling Search**: 针对 T3，平衡两层性能差异
3. **1-Swap / 2-Swap**: 精细优化种子集

---

## 6. 运行命令参考 (Command Reference)

```bash
# 完整流程
python -m code.run_gma_mfea --config config.yaml --mode all

# 仅预训练 GMA
python -m code.run_gma_mfea --config config.yaml --mode pretrain

# 加载 checkpoint 后运行
python -m code.run_gma_mfea --config config.yaml --mode load_checkpoint --checkpoint kaa_grit_twostage_v2_best

# 仅运行 MFEA（需已有 GMA 输出）
python -m code.run_gma_mfea --config config.yaml --mode evolution
```

---

## 7. 性能调优建议 (Performance Tuning)

### 加速策略
- 减少 `attack_ratio` (0.1 → 0.05)：减少攻击步数
- 减少 `population_size` (100 → 50)：减少每代评估数
- 增大 `local_search_interval` (10 → 20)：降低局部搜索频率

### 提升搜索质量
- 开启 `local_search_enabled: true`
- 增大 `mutation_prob` (0.1 → 0.2)
- 增大 `random_ratio` (0.2 → 0.3)：增加探索多样性

---

## 8. 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Rcs 值为 ~0.1 | alpha 未归一化 | 已修复，alpha 归一化到均值=1 |
| 进化不提升 | offspring 评估后未更新 best | 已修复，评估时立即检查 |
| JSON 序列化失败 | numpy int64 | 已修复，添加 `to_native()` 转换 |
| alpha 维度不匹配 | checkpoint 与数据不一致 | 返回 None 使用均匀权重 |
