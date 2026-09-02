---
name: vast-lifecycle
description: Use when the user pastes a vast.ai SSH command for a fresh rented GPU box (they rent then destroy repeatedly to save money). Runs the full ephemeral-box lifecycle - connect, clone, setup, pull trained ckpts from HF to skip retraining, hand off to the run, then push artifacts to HF before destroy. Wraps the aegis-setup skill.
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

3. **Skip the 6 GPU-h retrain — pull trained ckpts from HF.** The nudity ckpts
   are already trained and live in the private HF repo `Noridom1/aegis-ckpts`.
   Pull them ON THE BOX (box net ~20 MB/s beats laptop upload ~1 MB/s — never scp
   these from the laptop):
   ```bash
   $SSH 'cd /workspace/security-project \
     && mkdir -p results/results_with_AEGIS/AEGIS/models \
     && cd results/results_with_AEGIS/AEGIS/models \
     && hf download Noridom1/aegis-ckpts Diffusers-UNet-full-nudity-epoch_999.pt \
          --repo-type model --local-dir .'
   ```
   `hf` needs the write/read token — `bootstrap` already exported `HF_TOKEN` from
   `setup/.env`. The `Diffusers-*.pt` is all Phases 2–4 need (gen/FID/CLIP + DMA
   attacks); only pull the Compvis one (same repo) if resuming training. Only
   nudity is on HF so far — Van Gogh / church still need Phase 1 on the box (and
   get pushed to HF at the end, see below).

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

Destroy wipes the box. Push everything irreplaceable **UP to HF from the box**
(box net ~20 MB/s, no laptop bottleneck), then destroy. The only expensive
artifacts are trained ckpts (~6 GPU-h each to remake); eval results are cheap but
push them too. Do NOT scp DOWN to the laptop — that's the ~1 MB/s path we're
avoiding.

```bash
$SSH 'cd /workspace/security-project && \
  # new-concept ckpts, one file each (Diffusers is enough; add Compvis only to resume train)
  hf upload Noridom1/aegis-ckpts \
    results/results_with_AEGIS/AEGIS/models/Diffusers-UNet-full-vangogh-epoch_999.pt \
    Diffusers-UNet-full-vangogh-epoch_999.pt --repo-type model && \
  # eval outputs: TAR the image dirs first — thousands of tiny PNGs blow past HFs
  # ~10k-files/folder guideline and rate-limit uploading one by one
  tar czf eval-nudity.tar.gz eval-artifacts/ files/results/ && \
  hf upload Noridom1/aegis-ckpts eval-nudity.tar.gz eval-nudity.tar.gz --repo-type model'
```

Verify AFTER the fact by pulling from HF at leisure — this does NOT block destroy.
Once the `hf upload` commits print their URLs (exit 0): **safe to destroy.** State
that plainly, then the user destroys. Ckpts/PNGs are gitignored — never commit
them to git; they live in the HF repo, not the tree.

## Money rules (no persistence changes the math)

- **Never pay to retrain.** Pull trained ckpts from HF (step 3). Retraining a
  concept you already have is ~6 GPU-h wasted.
- **Setup ~1 h is the floor** with no persistent volume — it is a download hour on
  a GPU box. Accept it; the lever is not repeating work *inside* the run.
- **Destroy promptly.** An idle box still bills. Pull + verify, then destroy.
- If the user ever attaches a real vast.ai persistent volume later, point
  `AEGIS_DATA_ROOT` at its mount and setup drops to ~1 min — revisit this then.
