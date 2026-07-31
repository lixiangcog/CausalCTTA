#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pseudo_labeling import (
    TARGET_MANIFESTS,
    load_mc_dropout_model,
    load_rgb_tensor,
    mc_dropout_probability,
    seed_everything,
)


class FundusDataset(Dataset):
    def __init__(self, root, rows, image_size=512):
        self.root = Path(root)
        self.rows = rows
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        domain, rel_image, rel_mask = self.rows[index]
        image = load_rgb_tensor(self.root / rel_image, self.image_size)
        mask = Image.open(self.root / rel_mask).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.asarray(mask)
        target = np.stack((mask < 255, mask == 0), axis=0)
        return image, torch.from_numpy(target), domain


def dice_per_sample(prediction, target):
    intersection = (prediction & target).sum(dim=(2, 3), dtype=torch.float64)
    denominator = prediction.sum(dim=(2, 3), dtype=torch.float64) + target.sum(
        dim=(2, 3), dtype=torch.float64
    )
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--csv-names", nargs="+")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3377)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(args.seed)

    dataset_root = Path(args.dataset_root)
    csv_names = args.csv_names or TARGET_MANIFESTS
    rows = []
    for csv_name in csv_names:
        frame = pd.read_csv(dataset_root / csv_name)
        domain = csv_name.rsplit("_", 1)[0]
        rows.extend(zip([domain] * len(frame), frame["image"], frame["mask"]))

    loader = DataLoader(
        FundusDataset(dataset_root, rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = load_mc_dropout_model(args.model_file, device)

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ]
    totals = {threshold: torch.zeros(2, dtype=torch.float64) for threshold in thresholds}
    domain_totals = {
        threshold: defaultdict(lambda: torch.zeros(2, dtype=torch.float64))
        for threshold in thresholds
    }
    counts = defaultdict(int)

    with torch.no_grad():
        for batch_index, (images, target, domains) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            probability = mc_dropout_probability(model, images, args.passes)
            target = target.to(device, non_blocking=True).bool()

            for threshold in thresholds:
                scores = dice_per_sample(probability >= threshold, target).cpu()
                totals[threshold] += scores.sum(dim=0)
                for item_index, domain in enumerate(domains):
                    domain_totals[threshold][domain] += scores[item_index]
            for domain in domains:
                counts[domain] += 1

            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                done = min((batch_index + 1) * args.batch_size, len(rows))
                print(f"Progress: {done}/{len(rows)}", flush=True)

    print("\nOverall threshold sweep")
    for threshold in thresholds:
        mean = totals[threshold] / len(rows)
        print(
            f"threshold={threshold:.2f} disc={mean[0]:.4f} "
            f"cup={mean[1]:.4f} mean={mean.mean():.4f}"
        )

    print("\nPer-domain Cup Dice")
    for threshold in thresholds:
        values = []
        for domain in sorted(counts):
            score = domain_totals[threshold][domain][1] / counts[domain]
            values.append(f"{domain}={score:.4f}")
        print(f"threshold={threshold:.2f} " + " ".join(values))


if __name__ == "__main__":
    main()
