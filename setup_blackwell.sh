#!/usr/bin/env bash
# AEGIS env for Blackwell GPUs (RTX 5090 / sm_120).
#
# Why this exists: environment.yaml pins torch 1.11 / cudatoolkit 11.3, whose
# CUDA kernels stop at sm_86 (Ampere). sm_120 (Blackwell) gets
# "no kernel image is available for execution on the device". Blackwell needs
# torch >= 2.7 built for cu128, which in turn needs Python >= 3.10.
#
# This mirrors environment.yaml, changing ONLY what py3.10 / torch2.7 forces:
#   torch 1.11 cu113 -> 2.7.1 cu128        (Blackwell kernels)
#   python 3.8.5     -> 3.10               (torch 2.7 dropped 3.8)
#   pytorch-lightning 1.4.2 -> 1.9.5       (last 1.x; keeps the old import paths ldm uses)
#   numpy 1.19.2 -> 1.26.4                 (py3.10; stays on 1.x to avoid numpy-2 breakage)
#   omegaconf 2.1.1 -> 2.3.0              (py3.10)
# transformers 4.25.1 and diffusers 0.12.1 stay pinned: utils/text_encoder.py and
# utils/convertModels.py copy internals from exactly those versions.
#
#   bash setup_blackwell.sh          # build ./.venv, then: source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"

uv python install 3.10
uv venv --clear --python 3.10 .venv
source .venv/bin/activate

# torch FIRST, from the cu128 index (sm_120 kernels landed in 2.7)
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# albumentations (env yaml pins 0.4.3) dropped: its old setup.py won't build on py3.10,
# and AEGIS is retention-data-free so ldm.data / albumentations is never imported at train time.
# pin hf_hub to a version that still has cached_download (diffusers 0.12 imports it)
# and satisfies transformers 4.25 (<1.0).
uv pip install \
  numpy==1.26.4 \
  "huggingface_hub==0.13.4" \
  transformers==4.25.1 \
  diffusers==0.12.1 \
  pytorch-lightning==1.9.5 \
  omegaconf==2.3.0 \
  einops==0.3.0 \
  torchmetrics==0.11.4 \
  kornia==0.6 \
  opencv-python \
  invisible-watermark \
  imageio==2.9.0 imageio-ffmpeg==0.4.2 \
  torch-fidelity==0.3.0 \
  test-tube \
  ftfy regex tqdm \
  matplotlib wandb tabulate

# vendored deps: taming's top-level package ships NO __init__.py (a PEP-420 namespace
# package), so uv's PEP-660 editable install (find_packages-based) drops it and
# `import taming` fails. environment.yaml's old `-e git+...` worked only because legacy
# editable just puts the repo root on sys.path. Reproduce that with a .pth file.
mkdir -p src
[ -d src/taming-transformers ] || git clone --depth 1 https://github.com/CompVis/taming-transformers.git src/taming-transformers
[ -d src/CLIP ]               || git clone --depth 1 https://github.com/openai/CLIP.git src/CLIP
SP=$(python -c "import site; print(site.getsitepackages()[0])")
printf '%s\n%s\n' "$PWD/src/taming-transformers" "$PWD/src/CLIP" > "$SP/aegis_vendored.pth"
uv pip install --no-deps -e .

echo
python - <<PY
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
