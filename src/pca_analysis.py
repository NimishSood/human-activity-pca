"""Core PCA helpers used by the notebook."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_full_pca(scaled_features: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Fit PCA on the standardized feature matrix and keep the full variance profile."""
    pca_model = PCA(svd_solver="full")
    transformed = pca_model.fit_transform(scaled_features)

    component_numbers = np.arange(1, len(pca_model.explained_variance_ratio_) + 1)
    explained_variance = pd.DataFrame(
        {
            "component_number": component_numbers,
            "component": [f"PC{number}" for number in component_numbers],
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca_model.explained_variance_ratio_),
        }
    )

    pca_2d_scores = pd.DataFrame(
        transformed[:, :2],
        columns=["PC1", "PC2"],
        index=scaled_features.index,
    )

    return pca_model, explained_variance, pca_2d_scores


def build_component_summary(
    explained_variance: pd.DataFrame,
    component_numbers: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
) -> pd.DataFrame:
    """Keep a compact variance summary for presentation use."""
    requested = set(component_numbers)
    summary = explained_variance[explained_variance["component_number"].isin(requested)].copy()
    summary["explained_variance_ratio"] = summary["explained_variance_ratio"].round(4)
    summary["cumulative_explained_variance"] = summary["cumulative_explained_variance"].round(4)
    return summary.reset_index(drop=True)


def build_variance_target_summary(
    explained_variance: pd.DataFrame,
    targets: Iterable[float] = (0.90, 0.95),
) -> pd.DataFrame:
    """Find how many components are needed to reach selected variance thresholds."""
    rows: list[dict[str, float | int | str]] = []

    for target in targets:
        matching_rows = explained_variance[
            explained_variance["cumulative_explained_variance"] >= target
        ]
        if matching_rows.empty:
            continue

        first_match = matching_rows.iloc[0]
        rows.append(
            {
                "target_variance": f"{target:.0%}",
                "components_needed": int(first_match["component_number"]),
                "actual_cumulative_variance": round(float(first_match["cumulative_explained_variance"]), 4),
            }
        )

    return pd.DataFrame(rows)


def attach_metadata_to_scores(
    pca_scores: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join 2D PCA scores with activity labels and other metadata."""
    return pd.concat(
        [metadata.reset_index(drop=True), pca_scores.reset_index(drop=True)],
        axis=1,
    )


def get_top_feature_loadings(
    pca_model: PCA,
    feature_names: list[str],
    components: Iterable[int] = (1, 2),
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the strongest feature loadings for selected principal components."""
    rows: list[dict[str, float | str]] = []

    for component_number in components:
        component_index = component_number - 1
        component_name = f"PC{component_number}"
        loadings = pd.Series(
            pca_model.components_[component_index],
            index=feature_names,
            name=component_name,
        )

        strongest = (
            loadings.reindex(loadings.abs().sort_values(ascending=False).index)
            .head(top_n)
        )

        for feature_name, loading in strongest.items():
            rows.append(
                {
                    "component": component_name,
                    "feature_name": feature_name,
                    "loading": float(loading),
                    "absolute_loading": float(abs(loading)),
                }
            )

    loadings_frame = pd.DataFrame(rows)
    return loadings_frame.sort_values(["component", "absolute_loading"], ascending=[True, False]).reset_index(drop=True)


def summarize_activity_centroids(pca_scores_with_labels: pd.DataFrame) -> pd.DataFrame:
    """Compute average PC1 and PC2 positions for each activity class."""
    centroids = (
        pca_scores_with_labels.groupby("activity_label")
        .agg(
            sample_count=("activity_label", "size"),
            mean_pc1=("PC1", "mean"),
            mean_pc2=("PC2", "mean"),
        )
        .reset_index()
        .sort_values("activity_label")
        .reset_index(drop=True)
    )

    return centroids


def _categorize_feature_domain(feature_name: str) -> str:
    """Classify a HAR feature by its broad domain."""
    if feature_name.startswith("t"):
        return "Time domain"
    if feature_name.startswith("f"):
        return "Frequency domain"
    if feature_name.startswith("angle("):
        return "Angle / mixed"
    return "Other"


def _categorize_sensor_family(feature_name: str) -> str:
    """Classify a HAR feature by the main signal family it describes."""
    if feature_name.startswith("angle("):
        return "Angle / mixed"
    if "GravityAcc" in feature_name:
        return "Gravity acceleration"
    if "BodyGyro" in feature_name:
        return "Gyroscope"
    if "BodyAcc" in feature_name:
        return "Body acceleration"
    return "Other"


def _categorize_feature_style(feature_name: str) -> str:
    """Classify a HAR feature by the type of summary it captures."""
    if feature_name.startswith("angle("):
        return "Angle-based relationship"
    if "meanFreq()" in feature_name or "arCoeff()" in feature_name:
        return "Frequency / coefficient pattern"
    if "Mag" in feature_name or "sma()" in feature_name:
        return "Magnitude / overall intensity"
    if "Jerk" in feature_name:
        return "Jerk-related change"
    return "Statistical summary"


def summarize_loading_groups(
    pca_model: PCA,
    feature_names: list[str],
    components: Iterable[int] = (1, 2),
    top_n: int = 40,
) -> pd.DataFrame:
    """Summarize the strongest loadings by broad, presentation-friendly groups."""
    rows: list[dict[str, float | int | str]] = []

    grouping_functions = {
        "domain": _categorize_feature_domain,
        "sensor_family": _categorize_sensor_family,
        "feature_style": _categorize_feature_style,
    }

    for component_number in components:
        component_index = component_number - 1
        component_name = f"PC{component_number}"
        loadings = pd.Series(
            pca_model.components_[component_index],
            index=feature_names,
            name=component_name,
        )

        strongest = (
            loadings.abs()
            .sort_values(ascending=False)
            .head(top_n)
            .rename("absolute_loading")
            .reset_index()
            .rename(columns={"index": "feature_name"})
        )

        total_loading_mass = float(strongest["absolute_loading"].sum())

        for grouping_name, grouping_function in grouping_functions.items():
            grouped = (
                strongest.assign(group_name=strongest["feature_name"].apply(grouping_function))
                .groupby("group_name", as_index=False)
                .agg(
                    feature_count=("feature_name", "size"),
                    total_absolute_loading=("absolute_loading", "sum"),
                )
                .sort_values("total_absolute_loading", ascending=False)
                .reset_index(drop=True)
            )

            grouped["component"] = component_name
            grouped["grouping"] = grouping_name
            grouped["share_of_top_loading_mass"] = (
                grouped["total_absolute_loading"] / total_loading_mass
            ).round(4)

            rows.extend(grouped.to_dict("records"))

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    ordered_columns = [
        "component",
        "grouping",
        "group_name",
        "feature_count",
        "total_absolute_loading",
        "share_of_top_loading_mass",
    ]
    return summary[ordered_columns]


def build_component_profile(loading_group_summary: pd.DataFrame) -> pd.DataFrame:
    """Create a compact one-row profile for each component from grouped loadings."""
    if loading_group_summary.empty:
        return pd.DataFrame()

    records: list[dict[str, str | float]] = []

    for component in loading_group_summary["component"].drop_duplicates():
        component_rows = loading_group_summary[loading_group_summary["component"] == component]
        profile_row: dict[str, str | float] = {"component": component}

        for grouping in ("domain", "sensor_family", "feature_style"):
            top_group = (
                component_rows[component_rows["grouping"] == grouping]
                .sort_values("total_absolute_loading", ascending=False)
                .iloc[0]
            )

            profile_row[f"top_{grouping}"] = str(top_group["group_name"])
            profile_row[f"{grouping}_share"] = round(float(top_group["share_of_top_loading_mass"]), 4)

        records.append(profile_row)

    return pd.DataFrame(records)
