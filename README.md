# GMA-MFEA: Graph Manifold Alignment based Multifactorial Evolutionary Algorithm

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Working-green)](https://github.com/)

基于**图流形对齐**的多任务演化算法，用于求解**多层竞争网络下的鲁棒影响力最大化 (RCIM)** 问题。

> 详细技术架构请参考 [TECHNICAL_DOC.md](TECHNICAL_DOC.md)

---

## 📊 最新实验结果

在 475 节点双层网络上，budget=10：

| 任务 | Rcs 值 | 说明 |
|------|--------|------|
| T1 (Layer 1) | 57.37 | Layer 1 竞争影响力 |
| T2 (Layer 2) | 80.01 | Layer 2 竞争影响力 |
| T3 (联合) | 66.80 | 跨层协作鲁棒性 |

---

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n MIMMFEA python=3.8
conda activate MIMMFEA
pip install torch torch-geometric numpy networkx pyyaml tqdm
```

### 2. 运行方式

```bash
# 方式一：完整流程 (GMA预训练 + MFEA演化)
python -m code.run_gma_mfea --config config.yaml --mode all

# 方式二：加载已有checkpoint后运行
python -m code.run_gma_mfea --config config.yaml --mode load_checkpoint --checkpoint kaa_grit_twostage_v2_best

# 方式三：仅运行MFEA演化 (需已有GMA输出)
python -m code.run_gma_mfea --config config.yaml --mode evolution
```

---

## 📁 项目结构

```
GMA-MFEA/
├── config.yaml              # 主配置文件
├── code/
│   ├── run_gma_mfea.py      # 主入口
│   ├── gma_core/            # GMA 预训练引擎
│   │   ├── train_gma.py
│   │   ├── alignment.py
│   │   └── kaa_grit.py
│   ├── mfea_core/           # MFEA 演化引擎
│   │   ├── run_mfea.py
│   │   ├── tasks.py
│   │   ├── operators.py
│   │   ├── population.py
│   │   └── local_search.py
│   └── evaluation/          # 评估指标
│       ├── calc_metrics.py
│       └── approx_2hop.py
├── checkpoints/             # 预训练模型
├── data/alignment/          # GMA输出 (S_align, alpha)
├── dataset/train/           # 训练数据
└── results/                 # 实验结果
```

---

## ⚙️ 主要配置项

```yaml
mfea:
  population_size: 100      # 种群大小
  num_generations: 60       # 演化代数
  budget_l1: 10             # Layer 1 种子数
  budget_l2: 10             # Layer 2 种子数
  
  # 初始化比例
  elite_ratio: 0.25         # α采样
  robust_ratio: 0.25        # 2-hop鲁棒
  aligned_ratio: 0.20       # 跨层对齐
  random_ratio: 0.30        # 随机探索
  
  # 局部搜索
  local_search_enabled: true
  local_search_interval: 10

attack:
  attack_ratio: 0.1         # 攻击比例
  attack_mode: "2hop"
```

---

## 📈 输出解读

```
[Gen  10/60] Rcs: T1=57.37, T2=80.01, T3=66.80 | τ dist: T1=35, T2=32, T3=33 | 55.5s
```

| 字段 | 含义 |
|------|------|
| `Rcs T1/T2/T3` | 各任务的鲁棒竞争影响力 (平均每步影响节点数) |
| `τ dist` | 种群在三个任务上的分布 |

---

## 📚 文档

- [技术白皮书](TECHNICAL_DOC.md) - 详细架构、算法原理、实现细节
- [集成指南](KAA_GRIT_INTEGRATION.md) - KAA-GRIT 编码器集成说明

---

## 📝 License

此项目仅供学术研究使用。
