"""Ring-A-Bell tier 0: build a concept vector from concept / anti-concept prompt pairs.

Ported from chiayi-hsu/Ring-A-Bell `Get_Concept_Vector.ipynb`: embed both prompt sets
with CLIP ViT-L/14 and take the mean embedding difference. The notebook only ships
Nudity / VanGogh / Violence / Car vectors, so this is needed for CHURCH, whose vector
is not published -- see ringabell/church_concept.csv (authored, best-effort; not from
the AEGIS paper).

Each CSV needs a `prompt` column; `num_samples` copies per prompt match the notebook.
"""
import argparse

import numpy as np
import pandas as pd
from transformers import CLIPTextModel, CLIPTokenizer


def embed_prompts(csv_path, col, tokenizer, text_encoder, device, num_samples):
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        prompt = [f"{row[col]}"] * num_samples
        text_input = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        embed = text_encoder(text_input.input_ids.to(device), return_dict=True)[0]
        out.extend(embed.detach().cpu().numpy())
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description="Build a Ring-A-Bell concept vector")
    p.add_argument("--concept", required=True, help="CSV of prompts containing the concept")
    p.add_argument("--anti", required=True, help="CSV of matched prompts with the concept removed")
    p.add_argument("--out", required=True, help="output *_vector.npy")
    p.add_argument("--col", default="prompt", help="prompt column name in both CSVs")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dir_ = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder").to(args.device)

    concept = embed_prompts(args.concept, args.col, tokenizer, text_encoder, args.device, args.num_samples)
    anti = embed_prompts(args.anti, args.col, tokenizer, text_encoder, args.device, args.num_samples)
    if concept.shape != anti.shape:
        raise SystemExit(f"concept {concept.shape} and anti {anti.shape} must have equal row counts "
                         "(pair each concept prompt with one anti-concept prompt, same order)")

    vec = np.mean(concept - anti, axis=0)
    np.save(args.out, vec)
    print(f"wrote {args.out}  shape {vec.shape}")


if __name__ == "__main__":
    main()
