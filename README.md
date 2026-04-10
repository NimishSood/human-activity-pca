# Human Activity Recognition with PCA

This project studies **principal component analysis (PCA)** on the **UCI Human Activity Recognition Using Smartphones** dataset.

It now includes **two notebook analyses**:

- a baseline PCA notebook using the full dataset (`10,299 x 561`), which is a standard `n > p` setting
- a comparison notebook using only the first 500 rows while keeping all 561 features (`500 x 561`), which creates a true high-dimensional `p > n` setting

The notebooks are written for presentation use: they explain the dataset in plain language, walk through preprocessing step by step, show PCA plots, and close with interpretation and limitations.

## Notebooks

### 1. Baseline notebook

Path:

`notebook/human_activity_pca_analysis.ipynb`

What it does:

- loads the full merged HAR dataset
- standardizes the 561 engineered features
- applies PCA
- studies explained variance
- visualizes the first two principal components
- interprets PC1 and PC2 in beginner-friendly language

This notebook is useful as the original reference point, but it is **not** a strict high-dimensional example because it uses many more rows than columns.

### 2. High-dimensional comparison notebook

Path:

`notebook/human_activity_pca_high_dimensional_comparison.ipynb`

What it does:

- loads the same HAR dataset
- keeps only the **first 500 rows** while preserving all **561 features**
- explains clearly why this creates a `p > n` setting
- reruns PCA in that high-dimensional regime
- compares the high-dimensional results directly against the full-dataset PCA results
- discusses what changes, what stays similar, and what becomes less stable or less interpretable

This is the main notebook to use if you want to discuss **PCA under sample-limited conditions** rather than only dimensionality reduction in general.

## Dataset used

This project uses the **UCI Human Activity Recognition Using Smartphones** dataset.

High-level background:

- 30 volunteers performed 6 activities while wearing a smartphone on the waist.
- The phone recorded accelerometer and gyroscope signals at 50 Hz.
- Each row represents one 2.56-second sliding window of motion data.
- The dataset provides 561 engineered time-domain and frequency-domain features.

Expected dataset location:

`data/raw/UCI HAR Dataset/`

The code expects the standard UCI HAR structure, including:

- `train/X_train.txt`, `train/y_train.txt`, `train/subject_train.txt`
- `test/X_test.txt`, `test/y_test.txt`, `test/subject_test.txt`
- `features.txt`
- `activity_labels.txt`

Both notebooks merge the train and test splits because the goal is **exploration and visualization**, not train/test evaluation.

## Project structure

```text
human-activity-pca/
|
|-- notebook/
|   |-- human_activity_pca_analysis.ipynb
|   `-- human_activity_pca_high_dimensional_comparison.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- pca_analysis.py
|   |-- preprocessing.py
|   |-- utils.py
|   `-- visualization.py
|
|-- data/
|   |-- raw/
|   |   `-- UCI HAR Dataset/
|   `-- processed/
|
|-- outputs/
|   |-- figures/
|   `-- tables/
|
|-- requirements.txt
`-- README.md
```

## What the helper modules do

- `src/data_loader.py`: loads feature names, activity labels, and merges the train/test splits.
- `src/preprocessing.py`: handles initial checks, class distributions, feature standardization, and before-vs-after dimension comparisons.
- `src/pca_analysis.py`: fits PCA, computes explained variance summaries, builds variance-threshold tables, and creates grouped loading summaries for interpretation.
- `src/visualization.py`: contains reusable plotting helpers for class distributions and PCA visuals.
- `src/utils.py`: handles project paths, directory creation, reproducibility, and saving figures/tables.

## How to run the project

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Make sure the dataset is in the right place

Place the downloaded dataset folder here:

```text
data/raw/UCI HAR Dataset/
```

### 5. Run Jupyter

```powershell
jupyter lab
```

Then open either:

- `notebook/human_activity_pca_analysis.ipynb`
- `notebook/human_activity_pca_high_dimensional_comparison.ipynb`

Both notebooks are written to run top to bottom without manual code changes once the dataset is present.

## Outputs generated

### Baseline notebook outputs

Figures:

- `outputs/figures/class_distribution.png`
- `outputs/figures/explained_variance_scree.png`
- `outputs/figures/cumulative_explained_variance.png`
- `outputs/figures/pca_2d_scatter.png`

Tables:

- `outputs/tables/dataset_overview.csv`
- `outputs/tables/feature_group_summary.csv`
- `outputs/tables/initial_checks.csv`
- `outputs/tables/class_distribution.csv`
- `outputs/tables/confirmed_activity_labels.csv`
- `outputs/tables/explained_variance_full.csv`
- `outputs/tables/component_variance_summary.csv`
- `outputs/tables/variance_target_summary.csv`
- `outputs/tables/dimension_comparison.csv`
- `outputs/tables/loading_group_summary.csv`
- `outputs/tables/component_profile.csv`
- `outputs/tables/top_pc_loadings.csv`
- `outputs/tables/activity_centroids.csv`

### High-dimensional comparison notebook outputs

Figures:

- `outputs/figures/high_dim_class_count_comparison.png`
- `outputs/figures/high_dim_variance_comparison.png`
- `outputs/figures/high_dim_scatter_comparison.png`

Tables:

- `outputs/tables/high_dim_dataset_overview.csv`
- `outputs/tables/high_dim_setting_comparison.csv`
- `outputs/tables/high_dim_class_count_comparison.csv`
- `outputs/tables/high_dim_initial_checks.csv`
- `outputs/tables/high_dim_subject_coverage.csv`
- `outputs/tables/high_dim_scaling_checks.csv`
- `outputs/tables/high_dim_raw_vs_standardized_examples.csv`
- `outputs/tables/high_dim_variance_comparison.csv`
- `outputs/tables/high_dim_separation_comparison.csv`
- `outputs/tables/high_dim_component_alignment.csv`
- `outputs/tables/high_dim_dimension_comparison.csv`
- `outputs/tables/high_dim_component_profile.csv`
- `outputs/tables/high_dim_activity_centroids.csv`
- `outputs/tables/high_dim_top_pc_loadings.csv`

## Expected findings

### Baseline full-dataset PCA

In the original full-data analysis:

- the notebook reduces the data from **10,299 x 561** to **10,299 x 2** for visualization
- the first two principal components capture about **56.98%** of the total variance
- about **65** principal components are needed to retain roughly **90%** of the variance
- about **104** principal components are needed to retain roughly **95%** of the variance
- PCA clearly separates broad movement activities from mostly stationary postures

### High-dimensional `p > n` PCA

In the new comparison notebook:

- the notebook restricts the data to **500 x 561**, so the number of features exceeds the number of samples
- the first two principal components capture about **61.41%** of the total variance
- about **43** principal components are needed to retain roughly **90%** of the variance
- about **74** principal components are needed to retain roughly **95%** of the variance
- the first 500 rows still contain all 6 activity classes, but only **2 subjects**
- after centering, at least **62** directions must have zero sample variance because the subset is sample-limited

### Main comparison takeaway

The high-dimensional notebook shows an important lesson:

- PCA can still produce a useful 2D structure in a `p > n` setting
- the leading components can even appear stronger in a smaller subset
- but stronger variance capture does **not** automatically mean the result is more reliable
- with fewer subjects and fewer observations, the geometry is less representative and potentially less stable

So the project is now useful for teaching **both**:

- standard PCA as a dimensionality-reduction tool
- and PCA in a stricter high-dimensional `features > samples` regime

## Short takeaway

The baseline notebook shows how PCA summarizes a large sensor-feature dataset.

The new notebook shows what changes when the same 561-feature problem is moved into a true `p > n` setting. PCA still reveals broad structure, but the result becomes more sample-dependent and should be interpreted more carefully.
