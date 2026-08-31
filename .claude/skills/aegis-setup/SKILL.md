---
name: aegis-setup
description: Use when setting up, resuming, or debugging an AEGIS reproduce run on a new or rebuilt GPU box - bootstrapping the envs and 25 GB of assets, deciding which phases the machine can actually run, or diagnosing a failed setup/bootstrap.sh phase.
---

# AEGIS reproduce-box setup

Everything is already scripted. Your job is to run it, read what it reports, and
pick the right next phase — not to re-derive the setup.

## First move on any box

```bash
bash setup/bootstrap.sh --status     # what is already done
bash setup/bootstrap.sh              # do the rest; safe to re-run
```

Phases: `preflight secrets envs assets verify`. Each writes a marker to
`$AEGIS_DATA_ROOT/.bootstrap`, so a killed download resumes by re-running.
`--only <phase>`, `--skip <phase>`, `--force` to clear all markers.

If `setup/.env` is missing, `cp setup/env.example setup/.env` and fill in
`AEGIS_DATA_ROOT`, `HF_TOKEN`, optionally `WANDB_API_KEY` and `COCO_DRIVE_ID`.
Never commit `.env`, and never put a key in a tracked file (`AGENTS.md`).

## AEGIS_DATA_ROOT is the whole multi-machine trick

Point it at a volume that outlives the container (`/workspace` on RunPod, a
mounted disk elsewhere). `preflight` symlinks `models/`, `data/imgs/` and
`external/` there, so a rebuilt box re-clones the repo and finds the 25 GB
already present — setup drops from ~1 h to ~1 min. Getting this wrong is the
single biggest time sink, so confirm it before anything downloads.

## What the box can actually run

`preflight` prints VRAM and says so, but the rule is:

| VRAM | Runnable |
|---|---|
| ≥ 40 GB | everything (`GP` holds three extra per-parameter copies on top of Adam's two) |
| 24 GB | attacks + image generation; **not** training |
| < 16 GB | nothing |

Training is only ~6 GPU-h of the ~136 total, so a 24 GB box is still useful:
train elsewhere, copy the three `Diffusers-UNet-*.pt` files (3.4 GB each) in,
and run Phases 2–4 here. Checkpoints are the only artifact the two halves share.

## When a phase fails

| Symptom | Cause |
|---|---|
| `HF_TOKEN rejected (HTTP 401/403)` | licence not accepted on `CompVis/stable-diffusion-v-1-4-original` |
| `too small -- set AEGIS_DATA_ROOT` | pointed at the boot disk; needs ~100 GB |
| gdown returns an HTML page | Drive rate limit — retry later, or download by hand and drop it in place |
| `verify` says checkpoint-2800 missing | style_classifier.zip unpacked to a different layout; find the ViT dir and match `classifier_dir` in `configs/dma/aegis_vangogh_*.json` |
| `target_ckpt not found` from `run_attack.sh` | training has not produced `Diffusers-UNet-full-<concept>-epoch_999.pt` yet |

`python setup/check_config.py` runs with no GPU and no torch — use it to check
`AEGIS.py`'s flags after editing them.

## Then hand off

Setup done means `verify` passed. The run itself — phase order, budget, target
numbers, the μ / ω-clamp variants — is in `REPRODUCE.md`. Read it rather than
reconstructing the commands.
