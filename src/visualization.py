"""Plotting helpers for the PCA notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import save_figure

DEFAULT_ACTIVITY_ORDER = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def set_plot_style() -> None:
    """Apply a clean plotting style for notebook and saved figures."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150


def plot_class_distribution(
    class_distribution: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the number of examples in each activity class."""
    figure, axis = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=class_distribution,
        x="activity_label",
        y="count",
        hue="activity_label",
        order=DEFAULT_ACTIVITY_ORDER,
        palette="Set2",
        dodge=False,
        legend=False,
        ax=axis,
    )

    axis.set_title("Class Distribution Across the Six Activities")
    axis.set_xlabel("Activity")
    axis.set_ylabel("Number of windows")
    axis.tick_params(axis="x", rotation=25)

    if save_path is not None:
        save_figure(figure, save_path)

    return figure, axis


def plot_explained_variance(
    explained_variance: pd.DataFrame,
    save_path: str | Path | None = None,
    n_components: int = 20,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the variance explained by the first several principal components."""
    plot_data = explained_variance.head(n_components)
    figure, axis = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=plot_data,
        x="component_number",
        y="explained_variance_ratio",
        color="#3B82F6",
        ax=axis,
    )

    axis.set_title(f"Explained Variance for the First {n_components} Principal Components")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance ratio")

    if save_path is not None:
        save_figure(figure, save_path)

    return figure, axis


def plot_cumulative_explained_variance(
    explained_variance: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot cumulative explained variance across all PCA components."""
    figure, axis = plt.subplots(figsize=(11, 6))

    sns.lineplot(
        data=explained_variance,
        x="component_number",
        y="cumulative_explained_variance",
        color="#D97706",
        linewidth=2.5,
        ax=axis,
    )

    for target in (0.5, 0.8, 0.9):
        axis.axhline(target, linestyle="--", linewidth=1, color="gray", alpha=0.6)

    axis.set_title("Cumulative Explained Variance")
    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0, 1.02)

    if save_path is not None:
        save_figure(figure, save_path)

    return figure, axis


def plot_pca_scatter(
    pca_scores_with_labels: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the first two principal components and color points by activity."""
    figure, axis = plt.subplots(figsize=(11, 8))

    sns.scatterplot(
        data=pca_scores_with_labels,
        x="PC1",
        y="PC2",
        hue="activity_label",
        hue_order=DEFAULT_ACTIVITY_ORDER,
        palette="Set2",
        s=35,
        alpha=0.7,
        linewidth=0,
        ax=axis,
    )

    axis.set_title("2D PCA View of Smartphone Activity Data")
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.legend(title="Activity", bbox_to_anchor=(1.02, 1), loc="upper left")

    if save_path is not None:
        save_figure(figure, save_path)

    return figure, axis
