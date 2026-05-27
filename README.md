# Dense-Core Sparse-Edge Transformers

> **We study where MoE layers should be placed in Transformer language models and propose Dense-Core Sparse-Edge Transformers, which preserve dense attention and dense middle-layer FFNs while sparsifying only the input/output edge FFNs.**

一个可训练、可复现、可消融的研究原型。

## 核心假设

- **中间层承载共享语义表征** → 保持 Dense FFN
- **前缘层适合输入适配 / 后缘层适合输出专业化** → 只在边缘层使用 FFN-MoE
- **Dense Attention 全保留**，不做 Attention-MoE

## 架构

```
Layer 0-3:    Dense Attention + MoE-FFN  (前缘)
Layer 4-17:   Dense Attention + Dense FFN (核心)
Layer 18-23:  Dense Attention + MoE-FFN  (后缘)
```

每层结构：

```
x → RMSNorm → Dense Self-Attention → Residual
  → RMSNorm → FFN or MoE-FFN       → Residual
```

### MoE-FFN 设计

```text
MoEFFN(x) = shared_ffn(x) + weighted_sum(top_k routed_experts(x))
```

- 每个 expert 是 SwiGLU FFN: `W_down(SiLU(W_gate(x)) * W_up(x))`
- Token-level top-k routing
- Switch-style load balancing loss
- Shared expert 捕获公共能力，routed experts 负责边缘专业化

## 项目结构

```
project/
├── configs/                  # 配置文件
│   ├── dense_24l.yaml        # Dense baseline
│   ├── full_moe_24l.yaml     # 全层 MoE
│   ├── early_moe_24l.yaml    # 前缘 MoE
│   ├── middle_moe_24l.yaml   # 中间 MoE
│   ├── late_moe_24l.yaml     # 后缘 MoE
│   ├── edge_moe_24l.yaml     # 边缘 MoE（默认）
│   ├── random_moe_24l.yaml   # 随机 MoE
│   ├── edge_moe_10_80_10.yaml  # Core ratio ablation
│   ├── edge_moe_20_60_20.yaml
│   ├── edge_moe_30_40_30.yaml
│   ├── edge_moe_no_shared.yaml # Shared expert ablation
│   ├── edge_moe_top1.yaml      # Top-k ablation
│   ├── edge_moe_16e.yaml       # Expert count ablation
│   ├── dense_12l_125m.yaml     # 125M Dense for single GPU
│   ├── edge_moe_12l_125m.yaml  # 125M Edge-MoE for single GPU
│   └── sanity_tiny.yaml        # 快速测试
├── model/                    # 核心模型
│   ├── __init__.py
│   ├── attention.py          # DenseSelfAttention
│   ├── ffn.py                # DenseFFN / ExpertFFN (SwiGLU)
│   ├── moe.py                # MoEFFN + load_balancing_loss
│   ├── router.py             # TopKRouter
│   ├── placement.py          # get_moe_layers / is_moe_layer
│   ├── transformer.py        # 主模型 + RMSNorm + config
│   ├── metrics.py            # FLOPs 估算 / perplexity
│   └── upcycling.py          # Dense → MoE 初始化
├── infrastructure/           # 训练基础设施
│   ├── __init__.py
│   ├── metrics_logger.py     # JSONL 日志写入器
│   ├── run_manager.py        # 运行目录 & 元数据管理
│   └── checkpoint_manager.py # Checkpoint 保存/加载
├── analysis/                 # 分析 & 可视化工具
│   ├── __init__.py
│   ├── analyze_routing.py    # 路由分布分析
│   ├── analyze_route_consistency.py  # 路由一致性分析
│   ├── analyze_expert_specialization.py  # 专家特化 MI 分析
│   ├── summarize_runs.py     # 跨运行汇总对比
│   ├── plot_runs.py          # 训练曲线绘制（7 种图表）
│   └── estimate_flops.py     # FLOPs 估算
├── train.py                  # 训练脚本
├── eval.py                   # 评估脚本
├── utils.py                  # 工具函数
├── scripts/                  # Shell 脚本
│   ├── run_dense.sh
│   ├── run_edge_moe.sh
│   ├── run_ablation.sh       # 全量消融实验
│   └── run_tests.sh          # 单元测试 + sanity
├── tests/                    # 单元测试
│   ├── conftest.py
│   ├── test_moe_shapes.py
│   ├── test_router_topk.py
│   ├── test_load_balance_loss.py
│   ├── test_placement.py
│   ├── test_dense_attention_unchanged.py
│   ├── test_upcycling.py
│   └── test_experiment_system.py
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+
- PyTorch 2.x (CUDA 或 ROCm)
- PyYAML, pytest, datasets (可选)

### AMD ROCm 用户注意

AMD GPU 上 `scaled_dot_product_attention` 的 Flash/Mem-Efficient 模式默认是实验性的。如需启用以提升吞吐：

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

也可以忽略该 warning，PyTorch 会自动 fallback 到 math SDPA 实现。

### 已验证的硬件配置

| 硬件 | 状态 |
|------|------|
| AMD Radeon RX 7900 XT (21.5GB) + ROCm 6.3 | ✅ 通过测试 |
| NVIDIA GPU + CUDA | 应正常工作（未直接测试） |
| CPU-only | ✅ 通过 smoke test |

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行单元测试 (10 项全部通过)

```bash
pytest tests/ -v
```

### Sanity Check（超小模型快速验证，CPU 或 GPU 均可）

```bash
# CPU
CUDA_VISIBLE_DEVICES="" python train.py --config configs/sanity_tiny.yaml \
  --dataset random --tokens 1K --output_dir runs/sanity_cpu --batch_size 4 --seq_len 64

# GPU
python train.py --config configs/sanity_tiny.yaml \
  --dataset random --tokens 10K --output_dir runs/sanity_gpu --batch_size 16 --seq_len 64
```

### 125M 规模单卡实验

```bash
# Dense baseline (110M params)
python train.py --config configs/dense_12l_125m.yaml --dataset random --tokens 100M \
  --output_dir runs/dense_12l --batch_size 8 --seq_len 1024 --seed 42

# Edge-MoE (176M total / 119M active)
python train.py --config configs/edge_moe_12l_125m.yaml --dataset random --tokens 100M \
  --output_dir runs/edge_moe_12l --batch_size 8 --seq_len 1024 --seed 42
```

### 24 层完整训练

```bash
# Edge-MoE（默认配置）
python train.py --config configs/edge_moe_24l.yaml --dataset fineweb_sample \
  --tokens 1B --output_dir runs/edge_moe_24l

# Dense baseline
python train.py --config configs/dense_24l.yaml --dataset fineweb_sample \
  --tokens 1B --output_dir runs/dense_24l
```

### 评估

```bash
python eval.py --config configs/edge_moe_24l.yaml \
  --checkpoint runs/edge_moe_24l/checkpoint.pt --steps 20

python analysis/estimate_flops.py --config configs/edge_moe_24l.yaml
```

### 路由分析

```bash
python analysis/analyze_routing.py --trace runs/edge_moe_24l/routing_trace.jsonl
```

### Ubuntu中开启nvtop看显卡占用
```bash
sudo apt update && sudo apt install nvtop

nvtop
```

## 配置文件

配置文件通过 YAML 控制所有架构参数：

```yaml
model:
  architecture: dense_core_sparse_edge
  hidden_size: 768
  num_layers: 24
  num_attention_heads: 12
  intermediate_size: 3072          # Dense FFN 中间维度
  vocab_size: 32000
  max_position_embeddings: 2048

  attention:
    type: dense                    # 仅支持 dense

  ffn:
    placement: edge_moe            # dense / full_moe / early_moe / middle_moe
                                   # late_moe / edge_moe / random_moe
    front_moe_layers: 4
    back_moe_layers: 6

  moe:
    num_experts: 8
    top_k: 2
    shared_expert: true
    expert_intermediate_size: 1536 # MoE expert 中间维度（可小于 Dense FFN）
    router_aux_loss_coef: 0.01
    capacity_factor: 1.25
    router_jitter_noise: 0.0
    drop_tokens: false
```

### 放置策略

| 策略 | 说明 | MoE 层位置 |
|------|------|-----------|
| `dense` | 纯 Dense | 无 |
| `full_moe` | 全层 MoE | 所有层 |
| `early_moe` | 前缘 MoE | 前 N 层 |
| `middle_moe` | 中间 MoE | 中间 N 层 |
| `late_moe` | 后缘 MoE | 后 N 层 |
| `edge_moe` | 边缘 MoE（默认） | 前 M 层 + 后 K 层 |
| `random_moe` | 随机 MoE | 随机 N 层 |

### 训练参数

```bash
python train.py \
  --config configs/edge_moe_24l.yaml \
  --dataset fineweb_sample \       # random | fineweb_sample | /path/to/file.txt
  --tokens 1B \                    # 1B = 10亿, 10M = 1000万
  --output_dir runs/my_run \
  --batch_size 2 \
  --seq_len 2048 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --warmup_steps 100 \
  --log_interval 10 \
  --routing_trace_interval 100 \   # 每 N 步记录路由轨迹
  --seed 0
```

## Upcycling

从 Dense checkpoint 初始化 MoE 模型：

```bash
# 1. 先训练一个 dense baseline
python train.py --config configs/dense_24l.yaml --dataset fineweb_sample --tokens 1B --output_dir runs/dense_24l

# 2. 从 dense checkpoint 初始化 edge_moe
python train.py \
  --config configs/edge_moe_24l.yaml \
  --upcycle_from_dense_checkpoint runs/dense_24l/checkpoint.pt \
  --upcycling_noise_std 0.001 \
  --dataset fineweb_sample \
  --tokens 1B \
  --output_dir runs/edge_moe_upcycled
```

初始化规则：
- Dense FFN 权重 → shared expert
- Dense FFN 权重 → 每个 routed expert（可选加 noise）
- Router 权重随机初始化

## 实验矩阵

### Phase 1: Placement Ablation

```bash
bash scripts/run_ablation.sh
```

比较 7 种策略：dense / full_moe / early_moe / middle_moe / late_moe / edge_moe / random_moe

### Phase 2: Core Ratio Ablation

- `edge_moe_10_80_10` — 10% front + 80% core + 10% back
- `edge_moe_20_60_20` — 20% front + 60% core + 20% back
- `edge_moe_30_40_30` — 30% front + 40% core + 30% back

### Phase 3: Shared Expert & Top-K Ablation

- `edge_moe_no_shared` — 移除 shared expert
- `edge_moe_top1` — top-1 routing（替代 top-2）
- `edge_moe_16e` — 加倍 experts 数量

## 日志输出

训练时每个 MoE 层输出：

```text
layer_0/expert_0_load  layer_0/expert_1_load  ...
layer_0/router_entropy  layer_0/aux_loss
step=100 loss=3.4567 lm_loss=3.2345 aux_loss=0.2222 ppl=25.38 tokens_sec=12345.6 gpu_memory_mb=5123.4
```

评估输出：

```text
validation_loss
validation_perplexity
tokens_sec
total_parameters
active_parameters
active_flops_per_token_estimate
gpu_memory_mb
```

## 预期结论

| 假设 | 预期结果 |
|------|---------|
| Edge-MoE > Dense at same active FLOPs | ✓ |
| Edge-MoE ≈ Full-MoE on perplexity | ✓ |
| Edge-MoE > Full-MoE on routing stability | ✓ |
| Middle-MoE worse than Edge-MoE | ✓ |
| Shared expert improves stability | ✓ |

## 引用

- Switch Transformer: [Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022](https://arxiv.org/abs/2101.03961)
- Sparse Upcycling: [Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", ICLR 2023](https://arxiv.org/abs/2212.05055)
- DeepSeekMoE: [Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models", ACL 2024](https://arxiv.org/abs/2401.06066)
- Drop-Upcycling: [Wu et al., "From Dense to Sparse: Drop-Upcycling for Training MoE Language Models", 2025](https://arxiv.org/abs/2502.20561)

## License

MIT
