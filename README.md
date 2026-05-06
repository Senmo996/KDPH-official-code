# KDPH: Absorbing Gradient Conflicts via Kent Distributions for Cross-Modal Hashing

<div align="center">

**[Paper]** | **[arXiv]**

*Modeling Semantic Variance via Kent Distributions for Cross-Modal Hashing*

</div>

## Overview

**KDPH** (Kent Distribution Proxy Hashing) is a cross-modal hashing framework that maps images and texts into a shared Hamming space for efficient large-scale retrieval. It leverages **Kent distributions** on the unit hypersphere to model semantic variance within each category, absorbing gradient conflicts between intra-modal and cross-modal objectives. Built on CLIP (ViT-B/32), it achieves state-of-the-art performance on COCO, MIRFLICKR-25K, and NUS-WIDE benchmarks across 16/32/64-bit hash codes.

### Key Ideas

- **Kent Proxy Loss**: Replaces standard von Mises-Fisher proxies with Kent distributions, which capture both a mean direction *and* elliptical anisotropy via learnable orthogonal axes — modeling intra-class semantic variance more faithfully.
- **Conflict-Absorbing Optimization**: The anisotropic proxy structure naturally absorbs gradients that would otherwise conflict between intra-modal consistency and cross-modal alignment.
- **Hungarian Noise Assignment**: Per-batch binary noise vectors are assigned to embeddings via the Hungarian algorithm (minimum-cost bipartite matching), pulling hash codes toward discrete {-1, 1} values.

## Architecture

```
Image ──► CLIP ViT-B/32 ──► HashLayer (Linear + tanh) ──┐
                                                         ├──► KentProxyLoss ──► Binary Hash Codes
Text ───► CLIP ViT-B/32 ──► HashLayer (Linear + tanh) ──┘
```

## Datasets

| Dataset | Classes | Train | Query | Retrieval |
|---------|---------|-------|-------|-----------|
| COCO | 80 | 10,000 | 5,000 | 117,218 |
| MIRFLICKR-25K | 24 | 10,000 | 5,000 | 17,115 |
| NUS-WIDE | 21 | 10,000 | 5,000 | 190,421 |

Data format: `.mat` files containing image indices, text captions, and multi-label annotations.

## Quick Start

### Environment

- Python 3.8+
- PyTorch 2.3+ (CUDA 11.8)

```bash
pip install -r requirements.txt
```

### Download Requirements

1. **CLIP checkpoint**: Download `ViT-B-32.pt` and place it at the repo root.
2. **Data**: Download the [baidu-clip-hash-dataset](https://github.com/swuxyj/DeepHash-pytorch) and place it under `data_clip/baidu-clip-hash-dataset/`.
3. **codetable.xlsx**: Already included at `models/KDPH/loss/codetable.xlsx`.

### Training

```bash
# COCO, 16 bits
python main.py --config-file configs/KDPH/config-coco-16.yaml

# Flickr, 64 bits
python main.py --config-file configs/KDPH/config-flickr-64.yaml

# NUS-WIDE, 32 bits
python main.py --config-file configs/KDPH/config-nuswide-32.yaml
```

### Background Run

```bash
nohup python main.py --config-file "configs/KDPH/config-coco-16.yaml" > logs/KDPH_coco_16.log 2>&1 &
```

### Distributed Training

```bash
export CUDA_VISIBLE_DEVICES=0,1
python main.py --config-file configs/KDPH/config-flickr-16.yaml --distribute
```

## Configuration

All hyperparameters are managed via YAML configs (`configs/KDPH/`). Key parameters:

| Parameter | Description |
|-----------|-------------|
| `model.arch` | Model class (registered via `@registry.register_model`) |
| `model.alpha` | Weight for semantic consistency regularization |
| `model.lambda_cm` | Weight for cross-modal InfoNCE loss |
| `model.noise` | Weight for the Hungarian noise loss |
| `model.hypseed` | Random seed for proxy initialization |
| `model.numclass` | Number of proxy classes (80/24/21 for COCO/Flickr/NUS-WIDE) |
| `run.arch` | Trainer class (registered via `@registry.register_runner`) |
| `run.output_dim` | Hash code length (16/32/64) |
| `optimizer.hyp.lr` | Learning rate for Kent proxy optimizer (AdamW) |

## Evaluation

Metrics are computed at each validation epoch:
- **mAP(I→T)**: Image-to-text retrieval
- **mAP(T→I)**: Text-to-image retrieval
- **mAP(I→I)**: Image-to-image retrieval
- **mAP(T→T)**: Text-to-text retrieval

Best models (highest mAP I→T and mAP T→I) are saved as `.pth` checkpoints and `.mat` hash code files.

## Code Structure

```
├── main.py                 # Entrypoint
├── configs/
│   ├── base.yaml           # Base config template
│   └── KDPH/               # Per-dataset per-bit configs (9 files)
├── common/
│   ├── register.py         # Global registry (model/dataset/runner/optimizer/tokenizer)
│   └── calc_utils.py       # mAP computation
├── models/
│   ├── base.py             # BaseModel (CLIP backbone loader, freeze/unfreeze)
│   ├── CLIP/               # CLIP ViT-B/32 + BPE tokenizer
│   ├── baseline/           # Baseline hashing model
│   └── KDPH/
│       ├── KDPH.py         # Main KDPH model + Hungarian noise generation
│       ├── hash/hash.py    # HashLayer (separate image/text linear + tanh heads)
│       └── loss/
│           ├── HyP.py      # KentProxyLoss, InfoNCE, JSD losses
│           └── codetable.xlsx
├── runners/
│   ├── base.py             # BaseTrainer (training loop, evaluation, checkpointing)
│   └── KDPH/runner.py      # KDPHTrainer (dual-optimizer setup, tanh assertion)
├── dataset/
│   ├── base.py             # BaseDataset
│   ├── builder.py          # DataLoader builder
│   └── transformer_dataset.py  # CLIP-style .mat dataset with augmentation
└── utils/
    ├── get_args.py         # CLI arguments
    ├── logger.py           # Color logger
    └── set_seed.py         # Reproducibility
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{kdph2025,
  title   = {Absorbing Gradient Conflicts: Modeling Semantic Variance via Kent Distributions for Cross-Modal Hashing},
  author  = {Zhu, Hengjie and others},
  year    = {2025},
}
```

## License

This project is released for academic research purposes.
