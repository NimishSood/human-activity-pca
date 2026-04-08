"""Preprocessing helpers for the PCA workflow."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = ["split", "subject_id", "activity_id", "activity_label"]


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns used as PCA inputs."""
    return [column for column in dataframe.columns if column not in METADATA_COLUMNS]


def build_dataset_checks_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for first-pass quality checks."""
    feature_columns = get_feature_columns(dataframe)

    checks = [
        ("Rows (activity windows)", len(dataframe)),
        ("Feature columns", len(feature_columns)),
        ("Unique subjects", dataframe["subject_id"].nunique()),
        ("Activity classes", dataframe["activity_label"].nunique()),
        ("Missing values in features", int(dataframe[feature_columns].isna().sum().sum())),
        ("Training rows", int((dataframe["split"] == "train").sum())),
        ("Test rows", int((dataframe["split"] == "test").sum())),
    ]

    return pd.DataFrame(checks, columns=["metric", "value"])


def get_class_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count how many rows belong to each activity class."""
    distribution = (
        dataframe.groupby(["activity_id", "activity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("activity_id")
        .reset_index(drop=True)
    )

    return distribution


def get_sorted_activity_labels(dataframe: pd.DataFrame) -> list[str]:
    """Return activity labels in the dataset's natural numeric order."""
    ordered = (
        dataframe[["activity_id", "activity_label"]]
        .drop_duplicates()
        .sort_values("activity_id")
    )
    return ordered["activity_label"].tolist()


def standardize_features(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize the feature matrix before PCA."""
    selected_columns = list(feature_columns) if feature_columns is not None else get_feature_columns(dataframe)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(dataframe[selected_columns])
    scaled_frame = pd.DataFrame(scaled_array, columns=selected_columns, index=dataframe.index)

    return scaled_frame, scaler


def prepare_pca_input(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """Return the standardized feature matrix and the feature column names."""
    feature_columns = get_feature_columns(dataframe)
    scaled_features, scaler = standardize_features(dataframe, feature_columns=feature_columns)
    return scaled_features, feature_columns, scaler


def build_dimension_comparison_table(
    original_matrix: pd.DataFrame,
    reduced_matrix: pd.DataFrame,
    *,
    original_label: str = "Original feature space",
    reduced_label: str = "Reduced PCA space",
) -> pd.DataFrame:
    """Build a compact before-vs-after table for dimension reduction."""
    original_rows, original_columns = original_matrix.shape
    reduced_rows, reduced_columns = reduced_matrix.shape

    return pd.DataFrame(
        [
            {
                "representation": original_label,
                "matrix_shape": f"{original_rows:,} x {original_columns:,}",
                "rows_observations": original_rows,
                "columns_features": original_columns,
                "meaning": "All activity windows with the full engineered feature set.",
            },
            {
                "representation": reduced_label,
                "matrix_shape": f"{reduced_rows:,} x {reduced_columns:,}",
                "rows_observations": reduced_rows,
                "columns_features": reduced_columns,
                "meaning": "The same activity windows shown using only two principal components.",
            },
        ]
    )
