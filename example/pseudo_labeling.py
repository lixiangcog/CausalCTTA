"""Shared utilities for MC-Dropout pseudo-label generation and calibration."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_ROOT = PROJECT_ROOT / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from networks.Dropout_ResUnet import DropResUnet


TARGET_MANIFESTS = (
    "REFUGE_train.csv",
    "REFUGE_test.csv",
    "ORIGA_train.csv",
    "REFUGE_Valid.csv",
    "Drishti_GS_train.csv",
    "Drishti_GS_test.csv",
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb_tensor(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    minimum = float(array.min())
    scale = max(float(array.max()) - minimum, 1e-8)
    return torch.from_numpy((array - minimum) / scale)


def load_mc_dropout_model(model_file, device):
    model = DropResUnet(
        resnet="resnet34",
        num_classes=2,
        pretrained=False,
        convert=False,
    ).to(device)
    state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state, strict=True)

    # SFDA-DPL uses training mode for pseudo-label generation. This enables
    # Dropout2d sampling and target-batch statistics in BatchNorm.
    model.train()
    return model


def mc_dropout_probability(model, images, passes):
    if passes < 1:
        raise ValueError("passes must be at least 1")

    probability_sum = torch.zeros(
        (images.shape[0], 2, images.shape[2], images.shape[3]),
        device=images.device,
    )
    for _ in range(passes):
        logits, _, _ = model(images)
        probability_sum.add_(torch.sigmoid(logits))
    return probability_sum / passes
