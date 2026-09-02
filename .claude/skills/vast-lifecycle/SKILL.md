---
name: vast-lifecycle
description: Use when the user pastes a vast.ai SSH command for a fresh rented GPU box (they rent then destroy repeatedly to save money). Runs the full ephemeral-box lifecycle - connect, clone, setup, push local ckpts up to skip retraining, hand off to the run, then pull artifacts and verify before destroy. Wraps the aegis-setup skill.
---

# vast.ai box lifecycle

The user rents a GPU box on vast.ai, works, then **destroys it** to stop paying.
Nothing on the box survives destroy — `/workspace` is an ephemeral overlay wiped
on recycle (confirmed: no persistent volume). So every box is a blank slate, and
the two jobs of this skill are: bring a blank box up **without paying to redo
anything avoidable**, and get every irreplaceable artifact **off** before it dies.

The actual setup is already scripted (`setup/bootstrap.sh`) and covered by the
`aegis-setup` skill. This skill is the SSH wrapper around it.

## The one thing you get from the user

An SSH line, e.g. `ssh -p 23069 root@180.189.55.43`. Everything else derives from
it. Pull out the port and host and reuse them for scp (note scp is `-P`, capital):

```bash
PORT=23069 ; HOST=root@180.189.55.43        # <- from the pasted ssh line
SSH="ssh -p $PORT $HOST"
SCP="scp -P $PORT"
```

If the user only says "new box, here's the ssh", that IS the trigger — run the
lifecycle. Do not ask which phase; read the box and decide (VRAM table below).

## Bring the box up

Run these over SSH, in order. Steps 1–2 are the unavoidable cost floor (~1 h of
downloads); step 3 is the money lever.

1. **Clone** (public repo, no credentials):
   ```bash
   $SSH 'git clone https://github.com/izahai/security-project.git /workspace/security-project'
   ```

2. **Secrets + setup.** Bootstrap needs `HF_TOKEN` (and optionally `WANDB_API_KEY`).
   `setup/.env` is gitignored, so it is NOT in the clone. Get it onto the box one
   of two ways:
   - if a local `setup/.env` exists, just push it up (fastest, no re-pasting):
     `$SCP setup/.env $HOST:/workspace/security-project/setup/.env`
   - else on the box: `cp setup/env.example setup/.env` and fill in `HF_TOKEN`.

   Then hand off to `aegis-setup`:
   ```bash
   $SSH 'cd /workspace/security-project && bash setup/bootstrap.sh'
   ```
   `AEGIS_DATA_ROOT` defaults to `/workspace/aegis-data` — fine here, nothing
   persists anyway. bootstrap is idempotent/marker-based; re-run if a download dies.

   > **One-time local speed win:** create a gitignored `setup/.env` on this
   > Windows machine once (`cp setup/env.example setup/.env`, add `HF_TOKEN`).
   > Then every future box is one `scp` instead of re-pasting the token. It never
   > gets committed (`.gitignore`), so it is safe and it stays local.

3. **Skip the 6 GPU-h retrain — push trained ckpts UP.** The nudity ckpts are
   already trained and verified local. Do NOT let the box retrain them:
   ```bash
   $SSH 'mkdir -p /workspace/security-project/results/results_with_AEGIS/AEGIS/models'
   $SCP results/results_with_AEGIS/AEGIS/models/Diffusers-UNet-full-nudity-epoch_999.pt \
        results/results_with_AEGIS/AEGIS/models/Compvis-UNet-full-nudity-epoch_999.pt \
        $HOST:/workspace/security-project/results/results_with_AEGIS/AEGIS/models/
   ```
   The `Diffusers-*.pt` is all Phases 2–4 need (gen/FID/CLIP + DMA attacks). Only
   push the Compvis one if resuming training. Only nudity is trained so far —
   Van Gogh / church still need Phase 1 on the box.

## What this box can run

`bootstrap` preflight prints VRAM; the rule (from `aegis-setup`):

| VRAM | Runnable |
|---|---|
| ≥ 32 GB | everything |
| 24 GB | attacks + generation for sure; training borderline (~26 GB peak) |
| < 16 GB | nothing |

Then hand off: the run order, budget and target numbers are in `REPRODUCE.md`.
Don't reconstruct commands — read it.

## Before destroy — this is irreversible

Destroy wipes the box. Pull everything irreplaceable DOWN and **verify by hash**
before the user clicks destroy. The only expensive artifacts are trained ckpts
(~6 GPU-h each to remake); eval results are cheap but pull them too.

```bash
# ckpts (only ones the box produced that you don't already have — e.g. new concepts)
$SCP $HOST:'/workspace/security-project/results/results_with_AEGIS/AEGIS/models/*.pt' \
     results/results_with_AEGIS/AEGIS/models/
# eval outputs
$SCP -r $HOST:/workspace/security-project/eval-artifacts/ ./eval-artifacts/

# verify byte-exact BEFORE destroy: md5 must match both ends
$SSH 'md5sum /workspace/security-project/results/results_with_AEGIS/AEGIS/models/*.pt'
md5sum results/results_with_AEGIS/AEGIS/models/*.pt
```

Only when every hash matches: **safe to destroy.** State that plainly, then the
user destroys. Ckpts/PNGs are gitignored — never commit them; they live only as
these local copies.

## Money rules (no persistence changes the math)

- **Never pay to retrain.** Push local ckpts up (step 3). Retraining a concept you
  already have is ~6 GPU-h wasted.
- **Setup ~1 h is the floor** with no persistent volume — it is a download hour on
  a GPU box. Accept it; the lever is not repeating work *inside* the run.
- **Destroy promptly.** An idle box still bills. Pull + verify, then destroy.
- If the user ever attaches a real vast.ai persistent volume later, point
  `AEGIS_DATA_ROOT` at its mount and setup drops to ~1 min — revisit this then.
