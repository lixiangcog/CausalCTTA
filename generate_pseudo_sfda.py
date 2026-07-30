#!/usr/bin/env python3
import argparse
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


class FundusImages(Dataset):
    def __init__(self, root, rows, image_size=512):
        self.root = Path(root)
        self.rows = rows
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        csv_name, row_index, rel_image = self.rows[index]
        image = load_rgb_tensor(self.root / rel_image, self.image_size)
        return image, csv_name, row_index, rel_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--disc-threshold", type=float)
    parser.add_argument("--cup-threshold", type=float)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3377)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    csv_map = {
        name: name.replace(".csv", "_pseudo.csv") for name in TARGET_MANIFESTS
    }
    frames = {}
    rows = []
    for csv_name in csv_map:
        frame = pd.read_csv(dataset_root / csv_name)
        frames[csv_name] = frame
        for row_index, rel_image in enumerate(frame["image"].tolist()):
            rows.append((csv_name, row_index, rel_image))

    loader = DataLoader(
        FundusImages(dataset_root, rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = load_mc_dropout_model(args.model_file, device)

    disc_threshold = (
        args.threshold if args.disc_threshold is None else args.disc_threshold
    )
    cup_threshold = (
        args.threshold if args.cup_threshold is None else args.cup_threshold
    )
    print(
        f"Thresholds: disc={disc_threshold:.4f}, cup={cup_threshold:.4f}",
        flush=True,
    )

    pseudo_paths = {name: [None] * len(frame) for name, frame in frames.items()}
    channel_positive = np.zeros(2, dtype=np.int64)
    total_pixels = 0

    with torch.no_grad():
        for batch_index, (images, csv_names, row_indices, rel_images) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            prediction = mc_dropout_probability(model, images, args.passes)
            hard = torch.empty_like(prediction, dtype=torch.bool)
            hard[:, 0] = prediction[:, 0].gt(disc_threshold)
            hard[:, 1] = prediction[:, 1].gt(cup_threshold)

            channel_positive += hard.sum(dim=(0, 2, 3)).cpu().numpy()
            total_pixels += hard.shape[0] * hard.shape[2] * hard.shape[3]

            for item_index in range(hard.shape[0]):
                disc = hard[item_index, 0].cpu().numpy()
                cup = hard[item_index, 1].cpu().numpy()
                encoded = np.zeros(disc.shape, dtype=np.uint8)
                encoded[disc] = 1
                encoded[cup] = 2

                rel_image = Path(rel_images[item_index])
                destination = (output_root / rel_image).with_suffix(".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(encoded).save(destination)
                csv_name = csv_names[item_index]
                row_index = int(row_indices[item_index])
                try:
                    csv_path = destination.relative_to(dataset_root)
                except ValueError as error:
                    raise ValueError(
                        "--output-root must be located inside --dataset-root so "
                        "the generated CSV files contain anonymous relative paths"
                    ) from error
                pseudo_paths[csv_name][row_index] = str(csv_path)

            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                completed = min((batch_index + 1) * args.batch_size, len(rows))
                print(f"Progress: {completed}/{len(rows)}", flush=True)

    for csv_name, output_name in csv_map.items():
        frame = frames[csv_name].copy()
        frame["pseudo_label"] = pseudo_paths[csv_name]
        frame.to_csv(dataset_root / output_name, index=False)
        print(f"Wrote {output_name}: {len(frame)} rows")

    fractions = channel_positive / max(total_pixels, 1)
    print(f"Positive fractions: disc={fractions[0]:.6f}, cup={fractions[1]:.6f}")


if __name__ == "__main__":
    main()
