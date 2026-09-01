# Reproducing the AEGIS rows of Tables 2, 3 and 4

Scope: the **AEGIS** row only — nudity (Table 2, n=142), Van Gogh (Table 3, n=50), Church
(Table 4, n=50). Baseline rows are cited from the paper and the
[UnlearnDiffAtk leaderboard](https://huggingface.co/spaces/Intel/UnlearnDiffAtk-Benchmark);
they are measured on the same published prompt sets, so they do not need re-running.

ASR1 (P4D), ASR2 (UnlearnDiffAtk), FID and CLIP are the main path (Phases 0–4). ASR3
(Ring-A-Bell) is a separate transfer-attack workflow with its own caveats — see the last
section.

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

# --- Phase 1: train, 3 concepts (~6 GPU-h total, ~26 GB VRAM peak -- 32 GB safe)
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
| Train × 3 concepts | 6 | **~26 GB peak**, 32 GB safe (paper used 1× A6000, 2.01 h/concept) |
| FID/CLIP generation + scoring | 8.5 | 12 GB |
| ASR1 + ASR2, all 3 concepts | 122 | 24 GB |
| | **~136** | |

~6 days on one GPU, ~1.5 days sharded over four. Each `attack_idx` is independent, so sharding is a
plain stride with no coordination.

Attacks are batch-1 and latency-bound, so clock and bandwidth matter more than peak FLOPS. An
RTX 4090 (24 GB) is the best value for Phases 2–3 and *may* train, but it is borderline — see below.
An L40S or A6000 (48 GB) does everything with room. Avoid a T4: it turns 136 GPU-h into roughly 400.

Where the training ~26 GB goes (fp32 throughout — there is no autocast or `.half()` in the training
path, and gradient checkpointing is already on via `use_checkpoint: True` in the CompVis yaml):

| | GB |
|---|---|
| `model`, trainable LatentDiffusion (UNet 859.5M + VAE 83.7M + CLIP 123.1M) | 4.3 |
| `model_orig`, frozen reference — `--devices` defaults to `0,0`, so same GPU | 4.3 |
| separate HF `CLIPTextModel` (both `CustomTextEncoder`s wrap one object) + `all_embeddings` | 0.6 |
| Adam m+v over the 859.5M UNet params (`full` selects `model.model.diffusion_model` only) | 6.9 |
| gradients | 3.4 |
| `GP.pre_g_hat` | 3.4 |
| activations (batch 1, checkpointed) + allocator slack | ~3 |

Cheapest split: rent one 48 GB card for the ~6 h of training, keep the three `Diffusers-*.pt` files
(3.4 GB each), then move to 24 GB cards for the ~130 h of attacks. If you only have 24 GB, try
training there first — it is close enough that it may just fit; the peak is the attack backward
inside `soft_prompt_attack_max`.

**Disk: provision 100 GB.** Roughly: conda env 12, SD v1.4 ckpt 7.7, HF diffusers cache 5, style
classifier 1.8, COCO reals 1.3, AEGIS checkpoints 23, attack images 10, FID/CLIP images 12. The 30k
generated images are disposable once FID and CLIP have been read off (−12 GB); attack images must
survive until Phase 4, since both aggregators walk `images/` alongside `log.json`.

## ASR3 (Ring-A-Bell) — Phase 5

Ring-A-Bell is a **transfer** attack: adversarial ("InvPrompt") prompts are found offline using
only the CLIP text encoder + a concept vector, then fed to the erased model. So it splits in two:

- **Tier 1 — InvPrompt (model-free).** A genetic search per concept, once, independent of the AEGIS
  checkpoint. Ported verbatim from `chiayi-hsu/Ring-A-Bell` (notebooks only, no CLI) into
  `ringabell/inverse_prompt.py`; the church concept vector, which Ring-A-Bell never published, is
  built by `ringabell/get_concept_vector.py` from authored prompt pairs.
- **Tier 2 — score.** Generate from the InvPrompt CSV with the **same** `Diffusers-*.pt` as FID/CLIP,
  then detect the concept. `jobs/run_ringabell.sh` + `jobs/score_ringabell.sh` reuse the nudity
  (`nudenet-classes.py`) and object (`imageclassify.py`) detectors already in `eval-scripts/`; Van
  Gogh gets a new `eval-scripts/vangogh_classify.py` loading the WikiArt ViT (`checkpoint-2800`)
  already fetched for ASR2.

```bash
# assets: setup/fetch_assets.sh clones Ring-A-Bell (concept vectors) and, if HF_TOKEN has access,
# the gated nudity InvPrompt into ringabell/invprompt/nudity.csv.

conda activate AEGIS   # tier 1 + tier 2 both run in the training env (CLIP + diffusers)

# tier 1: build church vector, then GA Van Gogh + church InvPrompts (~8 GPU-h; nudity was fetched)
python ringabell/get_concept_vector.py \
    --concept ringabell/church_concept.csv --anti ringabell/church_anticoncept.csv \
    --out ringabell/vectors/church_vector.npy
for C in vangogh church; do
  python ringabell/inverse_prompt.py --concept_vector ringabell/vectors/${C}_vector.npy \
      --anchor_prompts data/prompts/dma_${C}.csv --eta 3 --length 16 \
      --out ringabell/invprompt/${C}.csv
done

# tier 2: generate with the erased UNet + detect (<1 GPU-h, 242 images total)
for C in nudity vangogh church; do
  bash jobs/run_ringabell.sh $C
  bash jobs/score_ringabell.sh $C
done
```

**Caveats — this is the least reproducible column, read before trusting the numbers:**

- **`--eta 3 --length 16` (K=16, η=3) are our choice, not the paper's.** AEGIS never states any
  Ring-A-Bell hyperparameter (App. F.4 says "the **two** attacks" while listing three; its N/50-step/
  40-iter/lr-0.01 settings are P4D + UnlearnDiffAtk only). η=3 is the notebook's nudity default; there
  is no published η for style or objects. If numbers are off, sweep η∈{2,3,4} first.
- **The church concept vector is authored** (`ringabell/church_concept.csv` / `church_anticoncept.csv`)
  — Ring-A-Bell publishes no church vector or InvPrompt. This is the weakest link of the three.
- **Nudity denominator differs.** The gated Ring-A-Bell nudity set is its own size (~95), not the
  n=142 of ASR1/ASR2; `score_ringabell.sh` reports over that set's own count. Van Gogh / church reuse
  the n=50 DMA prompt sets as GA anchors, so those denominators stay 50.
- **ASR3 target numbers**: read them off the paper's Table 2/3/4 to fill the column — not transcribed
  here yet.
