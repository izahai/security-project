# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AEGIS (ICLR 2026): adversarial, **retention-data-free** concept erasure for Stable Diffusion v1.4. A research
fork of CompVis `latent-diffusion` with an ESD/AdvUnlearn-style erasure loop bolted on. Almost everything
outside `train-scripts/` is vendored upstream code that the AEGIS pipeline does not touch.

Commit/PR and style conventions live in `AGENTS.md`; this file covers commands and architecture.

## Environment & commands

```bash
conda env create -f environment.yaml   # pinned py3.8.5 / cuda11.3 / torch1.11; installs this repo as editable `latent-diffusion`
conda activate AEGIS
pip install -e .                       # re-run after touching setup.py

# Train (run from repo ROOT — see "Import vs. CWD" below). Needs models/sd-v1-4-full-ema.ckpt.
python train-scripts/AEGIS.py --attack_init random --attack_step 1 --prompt 'nudity' --train_method 'full'

# Utility eval: generate 10k COCO images from an erased ckpt, then score FID/CLIP
bash jobs/fid_10k_generate.sh
bash jobs/tri_quality_eval.sh

# Only existing test
python scripts/tests/test_watermark.py PATH_TO_IMAGE
```

There is no test suite, linter, or formatter. Both `jobs/*.sh` are multi-GPU dispatchers whose `DIRS=("")` /
`NUMS=("999")` arrays are **empty placeholders** — fill them with your run directory and epoch before use.
They poll `nvidia-smi` and require real NVIDIA GPUs.

Robustness (attack) evaluation is *not* in this repo — Step 2 of the README hands the AEGIS UNet to
[UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack) externally.

## Architecture

### Import vs. CWD (read this before running anything)

`ldm/` and `train-scripts/ldm/` are **byte-identical copies**. `train-scripts/AEGIS.py` does
`from utils.util import *` with no `sys.path` fixup, so Python puts `train-scripts/` at `sys.path[0]`:

- `import ldm...` from any `train-scripts/` entry point resolves to **`train-scripts/ldm/`**, not the root copy.
- Data/model/output paths are relative to **CWD**, which must be the repo root (`./data/prompts/...`,
  `./models/sd-v1-4-full-ema.ckpt`, `./results/...`).

So: always launch from the repo root, and when changing diffusion internals for training, edit
`train-scripts/ldm/` (the root `ldm/` only backs `main.py` and `scripts/`). Keep the two copies in sync if you
change shared code.

### The training loop (`train-scripts/AEGIS.py`)

Four objects are held simultaneously:

| Object | Role |
|---|---|
| `model` / `sampler` | trainable CompVis `LatentDiffusion` loaded from the `.ckpt` |
| `model_orig` / `sampler_orig` | frozen reference copy, provides target scores |
| `custom_text_encoder` | HF `CLIPTextModel` wrapped by `utils/text_encoder.py` so it accepts `inputs_embeds` — this is what lets continuous adversarial embeddings be fed as text conditioning |
| `custom_text_encoder_orig` | frozen twin, used to build the clean/`emb_p` conditionings |

Per iteration:

1. Every `--adv_prompt_update_step` iterations, `soft_prompt_attack_max` (`utils/attack_util.py`) runs an inner
   optimization producing **two** adversarial prompt pairs: a *max* pair (`adv_word_embd`, `adv_input_ids`,
   k = `--adv_prompt_num` tokens) and a *min* pair (`..._min`, k=1). The max pair becomes the erasure target
   `emb_p`; the min pair becomes the model-side conditioning `emb_n`. `--attack_init latest` warm-starts from
   the previous round's embeddings; `random` re-initializes.
2. `get_train_loss_retain` (`utils/get_loss.py`) computes the ESD objective:
   `MSE(e_n, e_0 - negative_guidance * (e_p - e_0))`, where `e_n` comes from the trainable model and
   `e_0`/`e_p`/`e_new` from the frozen one, on a latent partially denoised to a random timestep.
3. `GP.DGR` (the `GP` class at the top of `AEGIS.py`) does the retention. This is the "retention-data-free"
   part: the retention direction is `g_r = 2*(θ - θ_orig)` in **parameter space**, so no retain dataset is
   needed. When the erasure gradient conflicts (`g_e · g_r < 0`) it subtracts a per-parameter weighted
   projection `w * proj(g_e onto g_r)`; `w` is adapted by the sign of the dot product with the previous
   projected gradient. Cosine and mean `w` are logged to W&B each step.
4. `opt.step()` on the Adam over `parameters`.

`param_choices` (`utils/util.py`) turns `--train_method` into the parameter list and decides which network
trains: any method containing `text_encoder` trains CLIP layers (UNet is `eval()`ed, checkpoint saved via
`save_text_encoder`); everything else (`full`, `noxattn`, `xattn`, `selfattn`, `notime`, `xlayer`,
`selflayer`) trains UNet params and saves via `save_model`.

Adversarial prompt construction lives in `attack_util.py`: `split_embd`/`split_id` cut the tokenized prompt,
`construct_embd*`/`construct_id*` reassemble it per `--attack_type` (`prefix_k`, `suffix_k`, `replace_k`,
`mid_k`, `insert_k`, `add`, `per_k_words`), and `project`/`embedding_to_input_id` snap continuous embeddings
back to nearest-vocabulary token ids.

### GRP knobs

`GP` (in `AEGIS.py`) deviates from the paper in two places, both now CLI flags rather than constants:
`--gp_mu` (paper App. F.3 says 0.1; the released code had 0.2) and `--gp_w_cap` (Alg. 2 caps ω at 1; the
released code clamped at 1e6). Defaults follow the paper. To reproduce the released code's behaviour:
`--gp_mu 0.2 --gp_w_cap 1e6`. This is not cosmetic — Table 6 puts fixed ω=1 at 8.45% vs 14.08% ASR₂.

`GP` also keeps three per-parameter tensor lists (`parameter_orig`, `w`, `pre_g_hat`), which is why
`--train_method full` needs ~40 GB of VRAM.

### Outputs

`./results/results_with_AEGIS/AEGIS/models/Compvis-UNet-{train_method}-{concept}-epoch_{i}.pt` (plus a
`Diffusers-...` conversion via `utils/convertModels.py:savemodelDiffusers`), loss curve under `logs/`,
W&B run dir under `results/results_with_AEGIS/wandb_logs`. `results/`, `*.ckpt`, `*.png` are gitignored —
never commit weights or generated images.

### Evaluation path

`jobs/fid_10k_generate.sh` → `train-scripts/generate-example-img.py`: builds a **Diffusers** SD-v1-4 pipeline
and swaps in the trained `{--save_path}.pt` as the target UNet (or text encoder, via
`extract_text_encoder_ckpt` / `get_openai_diffuser_transformer` which remaps HF→OpenAI CLIP key names).
`jobs/tri_quality_eval.sh` → `train-scripts/img_retain_eval.py`: FID via `T2IBenchmark`, CLIP score via HF
`clip-vit-base-patch32`. Both write a `.txt` next to the image folder.

`eval-scripts/` holds standalone metric scripts inherited from the ESD codebase (NudeNet class counts,
ImageNet classification accuracy, LPIPS, style loss, SLD generation). They are **not** wired into `jobs/`.

## Known rough edges

These are real, in the committed code — expect to hit them:

- `--diffusers_config_path` defaults to `diffusers_unet_config.json`, which does not exist — and that is
  harmless. `savemodelDiffusers` assigns `config_file = diffusers_config_file` and never reads it; the
  unet config comes from `create_unet_diffusers_config(original_config)`, i.e. from the CompVis yaml.
- `--attack_step` does **not** control the inner attack iterations: the `soft_prompt_attack_max` call sites
  pass a hardcoded `1` in that position. The flag only advances the W&B `global_step` counter.
- `--dataset_retain`, `--retain_batch`, `--retain_train`, `--retain_step`, `--retain_loss_w` and
  `--warmup_iter` are parsed and threaded through but unused — the retain-loss branches in
  `get_train_loss_retain` are commented out and it returns `unlearn_loss` alone. Retention is `GP.DGR` only.
- `--attack_method` accepts `multi_pgd` and `free_at`, but only `pgd` and `fast_at` are implemented; the
  others hit a `ValueError`.
- `attack_util.py` and `get_loss.py` contain large commented-out variants of the attack/loss functions. When
  editing, confirm you are in the live one (`soft_prompt_attack_max`, `get_train_loss_retain`).
