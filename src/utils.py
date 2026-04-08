"""Utility helpers for project paths, reproducibility, and saving outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET_FOLDER = "UCI HAR Dataset"


def find_project_root(start_path: Path | None = None) -> Path:
    """Locate the project root by looking for the src/ and data/ folders."""
    start = (start_path or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Expected to find 'src/' and 'data/' "
        "in the current directory or one of its parents."
    )


def get_project_paths(project_root: Path | None = None) -> dict[str, Path]:
    """Return the main project paths used throughout the analysis."""
    root = project_root or find_project_root()

    return {
        "root": root,
        "notebook": root / "notebook",
        "src": root / "src",
        "data": root / "data",
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "outputs": root / "outputs",
        "figures": root / "outputs" / "figures",
        "tables": root / "outputs" / "tables",
    }


def ensure_project_directories(project_root: Path | None = None) -> dict[str, Path]:
    """Create the standard project folders if they do not already exist."""
    paths = get_project_paths(project_root=project_root)

    for key in ("notebook", "src", "data_raw", "data_processed", "outputs", "figures", "tables"):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def resolve_dataset_dir(
    dataset_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve the dataset folder path and validate that it exists."""
    if dataset_dir is None:
        dataset_path = get_project_paths(project_root=project_root)["data_raw"] / DEFAULT_DATASET_FOLDER
    else:
        dataset_path = Path(dataset_dir)

    dataset_path = dataset_path.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at '{dataset_path}'. Place the UCI HAR dataset "
            f"inside 'data/raw/{DEFAULT_DATASET_FOLDER}/' or pass a custom path."
        )

    return dataset_path


def set_random_seed(seed: int = 42) -> None:
    """Set the NumPy random seed for reproducible results."""
    np.random.seed(seed)


def save_table(dataframe: pd.DataFrame, output_path: str | Path, *, index: bool = False) -> Path:
    """Save a pandas DataFrame as a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=index)
    return path


def save_figure(figure: Any, output_path: str | Path) -> Path:
    """Save a Matplotlib figure to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    return path
