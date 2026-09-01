"""Van Gogh style detector for ASR3, mirroring imageclassify.py's output columns.

The nudity (nudenet-classes.py) and object (imageclassify.py) detectors are already in
this repo, but there is no local Van Gogh classifier. This loads the same WikiArt ViT
that ASR1/ASR2 use -- results/checkpoint-2800, fetched by setup/fetch_assets.sh -- as a
plain HF image classifier and writes `category_top{k}` columns keyed on case_number, so
jobs/score_ringabell.sh can count "Van Gogh in top-k" exactly like the object path.

Match the artist label loosely (substring "gogh") since the checkpoint's id2label spelling
(e.g. "Vincent_van_Gogh") is only known once the archive is unpacked.
"""
import argparse
import os

import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    p = argparse.ArgumentParser(description="Van Gogh style classification via WikiArt ViT")
    p.add_argument("--folder_path", required=True, help="folder of generated images")
    p.add_argument("--prompts_path", required=True, help="the InvPrompt CSV (has case_number)")
    p.add_argument("--classifier_dir", default="results/checkpoint-2800", help="WikiArt ViT checkpoint dir")
    p.add_argument("--save_path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    folder = args.folder_path
    save_path = args.save_path or f"{folder}/{os.path.basename(folder.rstrip('/'))}_vangogh.csv"

    processor = AutoImageProcessor.from_pretrained(args.classifier_dir)
    model = AutoModelForImageClassification.from_pretrained(args.classifier_dir).to(args.device).eval()
    id2label = model.config.id2label

    names = [n for n in os.listdir(folder) if n.endswith(".png") or n.endswith(".jpg")]
    imgs = [Image.open(os.path.join(folder, n)).convert("RGB") for n in names]

    cats = {f"top{k}": [] for k in range(1, args.topk + 1)}
    scores = {f"top{k}": [] for k in range(1, args.topk + 1)}
    for i in range(0, len(imgs), args.batch_size):
        batch = imgs[i : i + args.batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(args.device)
        with torch.no_grad():
            probs = model(**inputs).logits.softmax(1)
        p_top, ids = torch.topk(probs, args.topk, dim=1)
        for k in range(1, args.topk + 1):
            ids_k = ids[:, k - 1].cpu().numpy()
            cats[f"top{k}"].extend([id2label[int(x)] for x in ids_k])
            scores[f"top{k}"].extend(p_top[:, k - 1].cpu().numpy())

    case_numbers = [int(n.split("_")[0].replace(".png", "").replace(".jpg", "")) for n in names]
    out = {"case_number": case_numbers}
    for k in range(1, args.topk + 1):
        out[f"category_top{k}"] = cats[f"top{k}"]
        out[f"scores_top{k}"] = scores[f"top{k}"]

    df = pd.read_csv(args.prompts_path)
    df["case_number"] = df["case_number"].astype("int")
    merged = pd.merge(df, pd.DataFrame(out))
    merged.to_csv(save_path)
    print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
