"""Ring-A-Bell tier 1: genetic search for adversarial ("InvPrompt") prompts.

Ported verbatim from chiayi-hsu/Ring-A-Bell `InversePrompt.ipynb` (the fitness /
crossover / mutation trio and the per-anchor GA loop are unchanged); only the CLI,
the anchor-CSV loop and the output format are new. The attack is target-model-free:
it uses only the CLIP text encoder + a precomputed concept vector, so one InvPrompt
set tests any erased model. See REPRODUCE.md "ASR3".

Per anchor prompt it optimises a length-K token sequence whose CLIP embedding matches
`embed(anchor) + eta * concept_vector`, then decodes the tokens into an adversarial
prompt. Output CSV is written straight in the `case_number,prompt,evaluation_seed`
format that train-scripts/generate-example-img.py consumes.
"""
import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import CLIPTextModel, CLIPTokenizer


# --- GA operators, verbatim from InversePrompt.ipynb (targetEmbed/text_encoder/
#     device/length are module-level there; kept as closures here). ---
def make_ops(text_encoder, target_embed, device, length):
    def fitness(population):
        dummy_tokens = torch.cat(population, 0)
        dummy_embed = text_encoder(dummy_tokens.to(device))[0]
        losses = ((target_embed - dummy_embed) ** 2).sum(dim=(1, 2))
        return losses.cpu().detach().numpy()

    def crossover(parents, crossover_rate):
        new_population = []
        for i in range(len(parents)):
            new_population.append(parents[i])
            if random.random() < crossover_rate:
                idx = np.random.randint(0, len(parents), size=(1,))[0]
                # idx 0 is 49406 (BOS); crossover points are 1..length.
                crossover_point = np.random.randint(1, length + 1, size=(1,))[0]
                new_population.append(torch.concat((parents[i][:, :crossover_point], parents[idx][:, crossover_point:]), 1))
                new_population.append(torch.concat((parents[idx][:, :crossover_point], parents[i][:, crossover_point:]), 1))
        return new_population

    def mutation(population, mutate_rate):
        for i in range(len(population)):
            if random.random() < mutate_rate:
                idx = np.random.randint(1, length + 1, size=(1,))
                value = np.random.randint(1, 49406, size=(1))[0]  # meaningful token, avoid 0/49406/49407
                population[i][:, idx] = value
        return population

    return fitness, crossover, mutation


def new_individual(length):
    """BOS + `length` random meaningful tokens + EOS padding to 77 (verbatim init)."""
    ind = torch.concat(
        (
            torch.from_numpy(np.array([[49406]])),
            torch.randint(low=1, high=49406, size=(1, length)),
            torch.tile(torch.from_numpy(np.array([[49407]])), [1, 76 - length]),
        ),
        1,
    )
    assert ind.shape == (1, 77), ind.shape  # 77 = BOS + length + padding, always
    return ind


def search_one(text_encoder, tokenizer, anchor_prompt, concept_vec, device, args):
    text_input = tokenizer(anchor_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    target_embed = text_encoder(text_input.input_ids.to(device))[0] + args.eta * concept_vec.to(device)
    target_embed = target_embed.detach().clone()

    fitness, crossover, mutation = make_ops(text_encoder, target_embed, device, args.length)
    population = [new_individual(args.length) for _ in range(args.population)]
    best = None
    for step in range(args.generation):
        score = fitness(population)
        idx = np.argsort(score)
        population = [population[i] for i in idx][: args.population // 2]
        best = score[idx[0]]
        if step != args.generation - 1:
            population = mutation(crossover(population, args.crossover), args.mutate)
        if step % 50 == 0:
            print(f"  gen {step + 1}/{args.generation}  min loss {best:.3f}", flush=True)
    return tokenizer.decode(population[0][0][1 : args.length + 1]), float(best)


def main():
    p = argparse.ArgumentParser(description="Ring-A-Bell InvPrompt genetic search")
    p.add_argument("--concept_vector", required=True, help="path to *_vector.npy")
    p.add_argument("--anchor_prompts", required=True, help="CSV with case_number,prompt,evaluation_seed")
    p.add_argument("--out", required=True, help="output InvPrompt CSV")
    p.add_argument("--eta", type=float, default=3.0, help="concept-vector strength (cof); paper is silent for style/object")
    p.add_argument("--length", type=int, default=16, help="K meaningful tokens (77 -> set 75)")
    p.add_argument("--population", type=int, default=200)
    p.add_argument("--generation", type=int, default=3000)
    p.add_argument("--mutate", type=float, default=0.25)
    p.add_argument("--crossover", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dir_ = "CompVis/stable-diffusion-v1-4"  # CLIP ViT-L/14, same as the notebook
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder").to(args.device)
    concept_vec = torch.from_numpy(np.load(args.concept_vector))

    df = pd.read_csv(args.anchor_prompts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_number", "prompt", "evaluation_seed"])
        for _, row in df.iterrows():
            case = int(row.case_number) if "case_number" in df.columns else _
            seed = int(row.evaluation_seed) if "evaluation_seed" in df.columns else args.seed
            print(f"[case {case}] anchor: {str(row.prompt)[:60]}")
            inv_prompt, loss = search_one(text_encoder, tokenizer, str(row.prompt), concept_vec, args.device, args)
            print(f"  -> InvPrompt (loss {loss:.3f}): {inv_prompt}")
            writer.writerow([case, inv_prompt, seed])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
