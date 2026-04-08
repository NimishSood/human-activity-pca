"""Dataset loading helpers for the UCI Human Activity Recognition dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils import resolve_dataset_dir


def _make_feature_names_unique(feature_names: Iterable[str]) -> list[str]:
    """Ensure feature names stay unique if the source file contains repeats."""
    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for name in feature_names:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            unique_names.append(name)
        else:
            unique_names.append(f"{name}__{counts[name]}")

    return unique_names


def load_feature_names(dataset_dir: str | Path | None = None) -> list[str]:
    """Load the 561 feature names defined by the dataset."""
    root = resolve_dataset_dir(dataset_dir)
    features_path = root / "features.txt"

    features = pd.read_csv(
        features_path,
        sep=r"\s+",
        header=None,
        names=["feature_index", "feature_name"],
    )

    return _make_feature_names_unique(features["feature_name"].tolist())


def load_activity_labels(dataset_dir: str | Path | None = None) -> dict[int, str]:
    """Load the mapping from numeric activity id to readable activity label."""
    root = resolve_dataset_dir(dataset_dir)
    labels_path = root / "activity_labels.txt"

    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["activity_id", "activity_label"],
    )

    return dict(zip(labels["activity_id"], labels["activity_label"]))


def _load_split(
    dataset_dir: Path,
    split_name: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Load one split (train or test) and add split metadata."""
    split_dir = dataset_dir / split_name

    feature_frame = pd.read_csv(
        split_dir / f"X_{split_name}.txt",
        sep=r"\s+",
        header=None,
        names=feature_names,
    )
    activity_frame = pd.read_csv(
        split_dir / f"y_{split_name}.txt",
        sep=r"\s+",
        header=None,
        names=["activity_id"],
    )
    subject_frame = pd.read_csv(
        split_dir / f"subject_{split_name}.txt",
        sep=r"\s+",
        header=None,
        names=["subject_id"],
    )

    split_column = pd.DataFrame({"split": [split_name] * len(feature_frame)})
    return pd.concat([split_column, subject_frame, activity_frame, feature_frame], axis=1)


def load_har_dataset(dataset_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and merge the train and test files into one tidy DataFrame."""
    root = resolve_dataset_dir(dataset_dir)
    feature_names = load_feature_names(root)
    activity_labels = load_activity_labels(root)

    train_frame = _load_split(root, "train", feature_names)
    test_frame = _load_split(root, "test", feature_names)

    combined = pd.concat([train_frame, test_frame], axis=0, ignore_index=True)
    activity_label_column = pd.DataFrame(
        {"activity_label": combined["activity_id"].map(activity_labels)}
    )

    return pd.concat(
        [
            combined[["split", "subject_id", "activity_id"]].reset_index(drop=True),
            activity_label_column.reset_index(drop=True),
            combined[feature_names].reset_index(drop=True),
        ],
        axis=1,
    )


def build_feature_group_summary(feature_names: Iterable[str]) -> pd.DataFrame:
    """Summarize the major feature families for an easy dataset overview."""

    def describe_domain(name: str) -> str:
        if name.startswith("t"):
            return "Time domain"
        if name.startswith("f"):
            return "Frequency domain"
        return "Angle / other"

    def describe_signal_family(name: str) -> str:
        if name.startswith("angle("):
            return "Angle features"
        if "BodyAccJerkMag" in name:
            return "Body acceleration jerk magnitude"
        if "BodyGyroJerkMag" in name:
            return "Body gyroscope jerk magnitude"
        if "BodyAccMag" in name:
            return "Body acceleration magnitude"
        if "GravityAccMag" in name:
            return "Gravity acceleration magnitude"
        if "BodyGyroMag" in name:
            return "Body gyroscope magnitude"
        if "GravityAcc" in name:
            return "Gravity acceleration"
        if "BodyAccJerk" in name:
            return "Body acceleration jerk"
        if "BodyGyroJerk" in name:
            return "Body gyroscope jerk"
        if "BodyAcc" in name:
            return "Body acceleration"
        if "BodyGyro" in name:
            return "Body gyroscope"
        if "angle" in name:
            return "Angle features"
        return "Other"

    summary_frame = pd.DataFrame({"feature_name": list(feature_names)})
    summary_frame["domain"] = summary_frame["feature_name"].apply(describe_domain)
    summary_frame["signal_family"] = summary_frame["feature_name"].apply(describe_signal_family)

    summary = (
        summary_frame.groupby(["domain", "signal_family"])
        .size()
        .reset_index(name="feature_count")
        .sort_values(["domain", "feature_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return summary
