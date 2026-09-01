#!/usr/bin/env bash
# Fetch everything the AEGIS reproduce run needs. Idempotent: skips what is already there.
# Run from the repo root on the GPU VM. Needs ~30 GB free for the downloads alone.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gdown >/dev/null || pip install -q gdown

# --- 1. UnlearnDiffAtk harness. Supplies the P4D + UnlearnDiffAtk attacks,
#        check_asr.py / style_analysis.py, and the published prompt sets.
if [ ! -d external/Diffusion-MU-Attack ]; then
  git clone --depth 1 https://github.com/OPTML-Group/Diffusion-MU-Attack.git external/Diffusion-MU-Attack
fi

# --- 2. Prompt sets. These are the sets the paper's n=50 / n=142 numbers are measured on.
mkdir -p data/prompts
for f in coco_10k nudity vangogh church parachute tench garbage_truck; do
  src="external/Diffusion-MU-Attack/prompts/$f.csv"
  [ -f "$src" ] && cp -n "$src" "data/prompts/dma_$f.csv"
done

# --- 3. WikiArt style classifier (ViT-base). Table 3's detector.
#        Must unpack to results/checkpoint-2800 -- that is the literal path
#        configs/style/text_grad_esd_vangogh_classsifier.json puts in classifier_dir.
if [ ! -d external/Diffusion-MU-Attack/results/checkpoint-2800 ]; then
  ( cd external/Diffusion-MU-Attack
    gdown 1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po -O style_classifier.zip   # 904 MB
    unzip -q style_classifier.zip && rm style_classifier.zip )
fi
[ -d external/Diffusion-MU-Attack/results/checkpoint-2800 ] \
  || echo "WARN: expected results/checkpoint-2800 after unzip; check the archive layout" >&2

# --- 4. ESD / FMN reference checkpoints. Optional, but the cheapest way to prove
#        the eval half of the pipeline is right before trusting an AEGIS number.
if [ "${FETCH_BASELINES:-0}" = "1" ]; then
  mkdir -p external/Diffusion-MU-Attack/files/pretrained
  ( cd external/Diffusion-MU-Attack/files/pretrained
    gdown 1e5aX8gkC34YaHGR0S1-EQwBmUXiAPvpE -O object_ckpt.zip   && unzip -q object_ckpt.zip   && rm object_ckpt.zip
    gdown 1yeZNJ8MoHsisdZmt5lbnG_kSgl5xned0 -O others_ckpt.zip   && unzip -q others_ckpt.zip   && rm others_ckpt.zip )
fi

# --- 4b. Ring-A-Bell (ASR3). Concept vectors ship in the repo; nudity InvPrompts are
#         gated. Van Gogh + church InvPrompts are generated on the VM by
#         ringabell/inverse_prompt.py, and church's concept vector by get_concept_vector.py.
mkdir -p ringabell/vectors ringabell/invprompt
if [ ! -d external/Ring-A-Bell ]; then
  git clone --depth 1 https://github.com/chiayi-hsu/Ring-A-Bell.git external/Ring-A-Bell
fi
for v in Nudity VanGogh; do
  src="external/Ring-A-Bell/Concept Vectors/${v}_vector.npy"
  dst="ringabell/vectors/$(echo "$v" | tr 'A-Z' 'a-z')_vector.npy"
  [ -f "$src" ] && cp -n "$src" "$dst"
done

# Gated nudity InvPrompt. Needs HF_TOKEN whose account was granted access at
# huggingface.co/datasets/Chia15/RingABell-Nudity. Normalised to the columns
# generate-example-img.py expects. If unavailable, GA it via inverse_prompt.py instead.
if [ ! -f ringabell/invprompt/nudity.csv ]; then
  if [ -n "${HF_TOKEN:-}" ]; then
    python - <<'PY' || echo "SKIP: nudity InvPrompt download failed (access not granted?). Request it, or GA with inverse_prompt.py." >&2
import os, glob, pandas as pd
from huggingface_hub import snapshot_download
d = snapshot_download(repo_id="Chia15/RingABell-Nudity", repo_type="dataset",
                      token=os.environ["HF_TOKEN"], local_dir="external/RingABell-Nudity")
csvs = glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
if not csvs:
    raise SystemExit("no CSV in the gated dataset")
df = pd.read_csv(csvs[0])
col = next((c for c in df.columns if c.lower() in ("prompt", "text", "adv_prompt", "invprompt")), df.columns[-1])
out = pd.DataFrame({"case_number": range(len(df)), "prompt": df[col].astype(str), "evaluation_seed": [6666] * len(df)})
out.to_csv("ringabell/invprompt/nudity.csv", index=False)
print(f"wrote ringabell/invprompt/nudity.csv ({len(out)} prompts, denominator for ASR3 nudity)")
PY
  else
    echo "SKIP: nudity InvPrompt needs HF_TOKEN with access to Chia15/RingABell-Nudity (gated)." >&2
  fi
fi

# --- 5. SD v1.4 CompVis checkpoint for training. Needs a HF token with the
#        CompVis licence accepted.
mkdir -p models
if [ ! -f models/sd-v1-4-full-ema.ckpt ]; then
  : "${HF_TOKEN:?set HF_TOKEN (huggingface.co/settings/tokens) and accept the CompVis/stable-diffusion-v-1-4-original licence}"
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o models/sd-v1-4-full-ema.ckpt \
    https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
fi

# --- 6. COCO-10k real images: the FID reference set img_retain_eval.py defaults to.
#        Only the AEGIS README knows the Drive id; put it in COCO_DRIVE_ID.
if [ ! -d data/imgs/coco_10k ]; then
  if [ -n "${COCO_DRIVE_ID:-}" ]; then
    mkdir -p data/imgs
    gdown "$COCO_DRIVE_ID" -O coco_10k.zip && unzip -q coco_10k.zip -d data/imgs && rm coco_10k.zip
  else
    echo "SKIP: data/imgs/coco_10k missing. Set COCO_DRIVE_ID (id from the AEGIS README) and re-run." >&2
  fi
fi

echo
echo "done. present:"
for p in external/Diffusion-MU-Attack external/Diffusion-MU-Attack/results/checkpoint-2800 \
         external/Ring-A-Bell ringabell/vectors/vangogh_vector.npy \
         models/sd-v1-4-full-ema.ckpt data/imgs/coco_10k data/prompts/dma_church.csv; do
  [ -e "$p" ] && echo "  ok      $p" || echo "  MISSING $p"
done
