"""Dataset manifest configuration shared by experiment entry points."""

from pathlib import Path


DATASETS = (
    "RIM_ONE_r3",
    "REFUGE",
    "ORIGA",
    "REFUGE_Valid",
    "Drishti_GS",
)
REQUIRED_PSEUDO_COLUMNS = ("image", "mask", "pseudo_label")


def target_datasets(source_dataset):
    if source_dataset not in DATASETS:
        choices = ", ".join(DATASETS)
        raise ValueError(f"Unknown source dataset {source_dataset!r}; choose from {choices}")
    return [dataset for dataset in DATASETS if dataset != source_dataset]


def pseudo_manifests_for(dataset):
    if dataset == "REFUGE_Valid":
        return ["REFUGE_Valid_pseudo.csv"]
    if dataset == "ORIGA":
        return ["ORIGA_train_pseudo.csv"]
    return [f"{dataset}_train_pseudo.csv", f"{dataset}_test_pseudo.csv"]


def load_target_records(dataset_root, datasets):
    import pandas as pd

    dataset_root = Path(dataset_root)
    images, masks, pseudo_labels = [], [], []

    for dataset in datasets:
        for manifest_name in pseudo_manifests_for(dataset):
            manifest_path = dataset_root / manifest_name
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing pseudo-label manifest: {manifest_path}")

            frame = pd.read_csv(manifest_path)
            missing = [
                column for column in REQUIRED_PSEUDO_COLUMNS if column not in frame.columns
            ]
            if missing:
                raise ValueError(
                    f"{manifest_path} is missing required columns: {', '.join(missing)}"
                )

            images.extend(frame["image"].tolist())
            masks.extend(frame["mask"].tolist())
            pseudo_labels.extend(frame["pseudo_label"].tolist())

    return images, masks, pseudo_labels
