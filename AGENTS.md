# Repository Guidelines

## Project Structure & Module Organization

This repository implements AEGIS, a Stable Diffusion concept-erasure research workflow. Core latent-diffusion components live in `ldm/`; training-specific code and its supporting copy of `ldm` are under `train-scripts/`. Use `train-scripts/AEGIS.py` as the main training entry point. Inference helpers are in `scripts/`, evaluation programs in `eval-scripts/`, and multi-GPU shell workflows in `jobs/`. YAML model definitions belong in `configs/` and `models/**/config.yaml`. Prompt datasets are stored in `data/prompts/`; large checkpoints and generated images must remain untracked.

## Build, Test, and Development Commands

- `conda env create -f environment.yaml` creates the pinned Python 3.8/CUDA environment.
- `conda activate AEGIS` activates it; the environment installs this package in editable mode.
- `pip install -e .` refreshes the local `latent-diffusion` package after environment setup.
- `python train-scripts/AEGIS.py --attack_init random --attack_step 1 --prompt nudity --train_method full` runs the documented training example. Place SD v1.4 at `models/sd-v1-4-full-ema.ckpt` first.
- `bash jobs/fid_10k_generate.sh` generates evaluation images; `bash jobs/tri_quality_eval.sh` computes configured FID/CLIP metrics. Both expect NVIDIA GPUs and may require job-array values to be configured.
- `python scripts/tests/test_watermark.py PATH_TO_IMAGE` performs the existing watermark smoke check.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module names. Keep CLI options descriptive and underscore-separated (for example, `--attack_step`). Preserve nearby import ordering and formatting; no repository-wide formatter or linter is currently configured. Avoid unrelated formatting changes, especially in vendored or duplicated diffusion modules.

## Testing Guidelines

There is no automated test suite or coverage threshold yet. For behavioral changes, add focused tests under `scripts/tests/` named `test_<feature>.py`, document any required model/data fixtures, and run the affected training or evaluation command on a small input. Do not commit generated images, checkpoints, or result directories.

## Commit & Pull Request Guidelines

History currently contains only `init`, so no established commit convention exists. Use short, imperative subjects such as `Fix adversarial prompt update`. Keep commits scoped. Pull requests should explain the research or implementation change, list exact validation commands, note GPU/model requirements, link relevant issues, and include before/after metrics or sample outputs when behavior changes.

## Security & Configuration

Never commit credentials, W&B keys, licensed datasets, or model weights. Keep machine-specific paths out of shared YAML and scripts; expose them through CLI arguments or environment variables instead.
