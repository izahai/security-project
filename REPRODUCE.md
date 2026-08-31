# Reproducing the AEGIS rows of Tables 2, 3 and 4

Scope: the **AEGIS** row only — nudity (Table 2, n=142), Van Gogh (Table 3, n=50), Church
(Table 4, n=50). Baseline rows are cited from the paper and the
[UnlearnDiffAtk leaderboard](https://huggingface.co/spaces/Intel/UnlearnDiffAtk-Benchmark);
they are measured on the same published prompt sets, so they do not need re-running.

ASR1 (P4D) and ASR2 (UnlearnDiffAtk) plus FID and CLIP are covered here.
ASR3 (Ring-A-Bell) is not — see the last section for why.

## Where the pieces come from

| Piece | Source |
|---|---|
| Prompt sets (n=50 / n=142, with per-row seeds and guidance 7.5) | `OPTML-Group/Diffusion-MU-Attack`, `prompts/*.csv` |
| ASR1 / ASR2 attacks | same repo, `src/execs/attack.py` |
| ASR aggregation | same repo, `scripts/analysis/{check_asr,style_analysis}.py` |
| Style detector (ViT-base on WikiArt, App. F.4) | same repo's Drive: `style_classifier.zip`, 904 MB |
| Object detector (ImageNet ResNet-50) | torchvision weights, fetched on use |
| Nudity detector (NudeNet) | `pip install nudenet` |
| FID / CLIP | this repo, `train-scripts/img_retain_eval.py` |

`bootstrap.sh` is idempotent: each of its phases (`preflight secrets envs assets verify`) leaves a
marker, so a killed download resumes by re-running it. `--status` shows what is left. Set
`AEGIS_DATA_ROOT` to a volume that outlives the container — it symlinks `models/`, `data/imgs/` and
`external/` there, so a rebuilt box finds the 25 GB already downloaded.

The two halves run in **separate conda envs** and are joined by exactly one artifact: the
`Diffusers-UNet-*.pt` that `save_model` writes. Both `src/tasks/classifier_.py` (attacks) and
`train-scripts/generate-example-img.py` (image generation) load it straight into a diffusers
`UNet2DConditionModel`, so no conversion glue is needed.

## Runbook

```bash
# --- Phase 0: setup (CPU, ~1 h first time, ~1 min on a box whose volume persisted)
cp setup/env.example setup/.env && $EDITOR setup/.env    # AEGIS_DATA_ROOT, HF_TOKEN
bash setup/bootstrap.sh

# --- Phase 1: train, 3 concepts (~6 GPU-h total, needs >=40 GB VRAM)
for C in nudity "Van Gogh" church; do
  python train-scripts/AEGIS.py --prompt "$C" --train_method full \
      --attack_init random --attack_step 1 --save_interval 1000
done
```

`--save_interval 1000` is not optional in practice. Each save writes the full `LatentDiffusion`
state dict (~4.3 GB) **plus** its diffusers conversion (~3.4 GB); the default `200` fires five times
per concept, so leaving it alone costs **115 GB instead of 23 GB**.

```bash
# --- Phase 2: FID + CLIP (~8.5 GPU-h, 12 GB VRAM is enough)
bash jobs/fid_10k_generate.sh            # 10k COCO-prompted images per concept
bash jobs/tri_quality_eval.sh            # FID via T2IBenchmark, CLIP via clip-vit-base-patch32

# --- Phase 3: attacks (~122 GPU-h, 24 GB VRAM). Third arg = GPUs to shard over.
conda activate dma
bash jobs/run_attack.sh vangogh unlearndiff 4      # start here: cheapest concept
bash jobs/run_attack.sh vangogh P4D 4
bash jobs/run_attack.sh church no_attack 4         # Pre-ASR: check_asr.py needs it
bash jobs/run_attack.sh church unlearndiff 4
bash jobs/run_attack.sh church P4D 4
bash jobs/run_attack.sh nudity no_attack 4
bash jobs/run_attack.sh nudity unlearndiff 4       # 142 prompts, the expensive one
bash jobs/run_attack.sh nudity P4D 4

# --- Phase 4: aggregate (CPU, seconds)
bash jobs/score_asr.sh vangogh
bash jobs/score_asr.sh church
bash jobs/score_asr.sh nudity
```

Do Van Gogh first on purpose: 50 prompts, and it is the only concept that scores without a separate
`no_attack` run (`style_analysis.py` reads pre-ASR out of log entry 0). If its numbers land, the
pipeline is right and the 71 GPU-h for nudity is worth spending.

`run_attack.sh` skips any `attack_idx` that already has a `log.json`, so re-running it is the resume
path.

## Targets

| | ASR1 | ASR2 | FID | CLIP |
|---|---|---|---|---|
| Nudity (n=142) | 12.06 | 8.45 | 17.43 | 0.303 |
| Van Gogh (n=50) | 36 | 12 | 17.25 | 0.310 |
| Church (n=50) | 28 | 6 | 19.06 | 0.305 |

Two knobs to try if AEGIS misses these. The released `GP` code differs from the paper in μ (0.2 vs
App. F.3's 0.1) and in the ω clamp (1e6 vs Alg. 2's 1). Defaults here follow the paper; to get the
released behaviour pass `--gp_mu 0.2 --gp_w_cap 1e6`. Table 6 shows this range matters (fixed ω=1
gives 14.08% ASR₂ against the dynamic 8.45%), so run both on Van Gogh and keep whichever matches.

If AEGIS is off but you want to know which half is at fault, run a public ESD checkpoint
(`FETCH_BASELINES=1 bash setup/fetch_assets.sh`) through the same Phase 3–4 and compare against the
leaderboard. ESD matching means the eval half is sound and the problem is in training.

`--top_k` for the style detector is unspecified in the paper: the script defaults to 10, the upstream
README suggests `{1|3}`. `score_asr.sh` prints all three; the correct one is whichever puts base
SD v1.4 at 100%.

## Budget

| Phase | GPU-h | VRAM |
|---|---|---|
| Train × 3 concepts | 6 | **≥40 GB** (paper: 1× A6000, 2.01 h/concept) |
| FID/CLIP generation + scoring | 8.5 | 12 GB |
| ASR1 + ASR2, all 3 concepts | 122 | 24 GB |
| | **~136** | |

~6 days on one GPU, ~1.5 days sharded over four. Each `attack_idx` is independent, so sharding is a
plain stride with no coordination.

Attacks are batch-1 and latency-bound, so clock and bandwidth matter more than peak FLOPS. An
RTX 4090 (24 GB) is the best value for Phases 2–3 but **cannot train** — the `GP` class holds three
extra per-parameter copies on top of Adam's two. An L40S or A6000 (48 GB) does everything. Avoid a
T4: it turns 136 GPU-h into roughly 400.

Cheapest split: rent one 48 GB card for the ~6 h of training, keep the three `Diffusers-*.pt` files
(3.4 GB each), then move to 24 GB cards for the ~130 h of attacks.

**Disk: provision 100 GB.** Roughly: conda env 12, SD v1.4 ckpt 7.7, HF diffusers cache 5, style
classifier 1.8, COCO reals 1.3, AEGIS checkpoints 23, attack images 10, FID/CLIP images 12. The 30k
generated images are disposable once FID and CLIP have been read off (−12 GB); attack images must
survive until Phase 4, since both aggregators walk `images/` alongside `log.json`.

## ASR3 (Ring-A-Bell) — not covered

Deliberately deferred; it is the least reproducible column in the paper.

- The hyperparameters are never given. App. F.4 says "Settings for the **two** attacks are in
  Appx. F.4" while listing three, and the settings it does describe (N prepended tokens, 50
  timesteps, 40 iterations, AdamW lr 0.01) only apply to P4D and UnlearnDiffAtk. Ring-A-Bell is a
  genetic search with an entirely different parameter set (K, η, population, generations), none of
  which appears anywhere in the paper.
- `chiayi-hsu/Ring-A-Bell` ships notebooks only, no CLI.
- Its `Concept Vectors/` has Nudity, VanGogh, Violence and Car — **no Church**, so Table 4's ASR3
  needs a concept vector built from scratch via `Get_Concept_Vector.ipynb`.
- The nudity InvPrompts live in a gated HF dataset (`Chia15/RingABell-Nudity`) that requires an
  access request. Send it early if you intend to do this column.
