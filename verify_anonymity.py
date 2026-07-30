#!/usr/bin/env python3
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {".git", "__pycache__"}
IDENTITY_PATTERNS = {
    "email address": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    "home-directory path": re.compile(r"/(?:home|Users|data/user)/[^/\s]+/"),
}
ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".bin",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".tar",
    ".tgz",
    ".7z",
    ".rar",
    ".zip",
}
DATA_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".tif", ".tiff", ".png"}
ALLOWED_BINARY_PATHS = {Path("image/Figure.png")}
FORBIDDEN_DIRECTORY_NAMES = {
    "checkpoints",
    "data",
    "datasets",
    "lightning_logs",
    "logs",
    "mlruns",
    "models",
    "runs",
    "wandb",
    "weights",
}
MAX_RELEASE_FILE_BYTES = 5 * 1024 * 1024


def main():
    findings = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRECTORY_NAMES:
                findings.append(f"experiment artifact directory: {relative}")
            continue
        if path.stat().st_size > MAX_RELEASE_FILE_BYTES:
            findings.append(f"unexpected large file: {relative}")
        if path.suffix.lower() in ARTIFACT_SUFFIXES:
            findings.append(f"artifact file: {relative}")
        if (
            path.suffix.lower() in DATA_IMAGE_SUFFIXES
            and relative not in ALLOWED_BINARY_PATHS
        ):
            findings.append(f"dataset or generated image: {relative}")
        lowered_name = path.name.lower()
        if lowered_name == "results.txt" or lowered_name.startswith("slurm-"):
            findings.append(f"experiment record: {relative}")
        if lowered_name.startswith("events.out.tfevents") or lowered_name.endswith(".log"):
            findings.append(f"experiment log: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in IDENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{label}: {relative}:{line}")

    if findings:
        print("Anonymity verification failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("Anonymity verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
