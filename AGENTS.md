# KDPH — Cross-Modal Hashing with Kent Distributions

## One-line description
PyTorch research code for cross-modal hashing using CLIP backbone + Kent proxy loss.

## Entrypoint
- `python main.py --config-file <config>` — training/eval entrypoint
- All imports are side-effect-driven: `from dataset import *`, `from models import *`, `from runners import *` trigger `@registry.register_*()` decorators so the registry is populated before `main()` runs.

## Config system
- YAML-based via `omegaconf`
- Configs: `configs/KDPH/config-{dataset}-{bits}.yaml` where dataset ∈ {coco, flickr, nuswide}, bits ∈ {16,32,64}
- Required keys: `model.arch` (maps to `@registry.register_model`), `run.arch` (maps to `@registry.register_runner`), plus dataset/optimizer configs
- Model params (`alpha`, `lambda_cm`, `noise`, `hypseed`, `numclass`) are tuned per dataset — DO NOT assume defaults work

## Training & running
```
python main.py --config-file configs/KDPH/config-coco-16.yaml
```
- Default device: CPU (CUDA_VISIBLE_DEVICES not set by default). Set via `run.device` in config or `--device` arg.
- Distributed training: `--distribute` flag + `CUDA_VISIBLE_DEVICES` env + device IDs in config
- Background run: `nohup python main.py --config-file "configs/KDPH/config-flickr-64.yaml" > logs/KDPH_flickr_64.log 2>&1 &`
- Run from repo root; `cd ../..` is relative to `runners/KDPH/` per README

## Required data/model files
- `data_clip/baidu-clip-hash-dataset/{coco,flickr,nuswide}/mat/` — expects `.mat` files (`caption.mat`, `index.mat`, `label.mat`)
- `./ViT-B-32.pt` — CLIP ViT-B/32 checkpoint (not in repo; must be downloaded externally)
- `models/KDPH/loss/codetable.xlsx` — Excel file read by KDPH model at init (must exist)

## Architecture
- `models/` — model definitions, `@registry.register_model("name")` decorator
- `runners/` — training loops inheriting `BaseTrainer`, `@registry.register_runner("name")` decorator
- `dataset/` — dataset loading from `.mat` files, `@registry.register_dataset("name")`
- `common/register.py` — global `registry` singleton; lookups done by string key from YAML `arch` fields
- `utils/` — arg parsing, logger, seed setting
- Model `KDPH` loads CLIP backbone, attaches `HashLayer`, and a `KentProxyLoss` module (`self.hyp`) trained with separate AdamW optimizer

## Logging & metrics
- **wandb** — auto-initialized; project name `clip-hash`, run name `{dataset}-{model.arch}`. Disable by defaulting `os.environ["WANDB_MODE"] = "disabled"` if W&B breaks.
- **Best model saved** when `mAPi2t` improves; saves `.pth` + `.mat` (hash codes + labels)
- Mat files saved to `{save_dir}/mat_files/{i2t-best|t2i-best|last}.mat`
- Keys: `q_img`, `q_txt`, `r_img`, `r_txt`, `q_l`, `r_l`

## Key conventions
- Binary hash codes are in {-1, 1} (sign of tanh output); `make_hash_code()` calls `sign_()`
- `hyp` submodule optimizer is `AdamW` (not BertAdam like the main model)
- Hash function must be `tanh` — assertion at runner init
- Hungarian algorithm (`linear_sum_assignment`) used in noise generation per batch
- `model.freezen()` / `model.unfreezen()` called in `change_state()` — these must exist on the model