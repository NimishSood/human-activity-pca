# Human Activity Recognition with PCA

This mini project explores **dimension reduction using PCA** on the **UCI Human Activity Recognition Using Smartphones** dataset.

The project is organized for a class presentation: the notebook tells the story, and the helper modules keep the code readable.

## What this project does

The notebook:

- introduces the UCI HAR dataset in simple language,
- explains why dimension reduction is useful when there are 561 features,
- loads and checks the dataset,
- standardizes the features before PCA,
- shows the concrete reduction from the original feature space to a 2D PCA view,
- applies PCA and measures explained variance,
- reports how many principal components are needed to retain about 90% and 95% of the variance,
- visualizes the first two principal components,
- interprets what PC1 and PC2 seem to represent in plain English,
- explains what the 2D PCA view does and does not show.

The main analytical question is:

> Can PCA compress the high-dimensional smartphone activity dataset into a small number of components while still preserving enough structure to visually distinguish major human activities?

## Dataset used

This project uses the **UCI Human Activity Recognition Using Smartphones** dataset.

High-level background:

- 30 volunteers performed 6 activities while wearing a smartphone on the waist.
- The phone recorded accelerometer and gyroscope signals at 50 Hz.
- Each row represents one 2.56-second sliding window of motion data.
- The dataset provides 561 engineered time-domain and frequency-domain features.

Expected dataset location inside this project:

`data/raw/UCI HAR Dataset/`

The code is written around the standard UCI HAR folder structure, including:

- `train/X_train.txt`, `train/y_train.txt`, `train/subject_train.txt`
- `test/X_test.txt`, `test/y_test.txt`, `test/subject_test.txt`
- `features.txt`
- `activity_labels.txt`

For this analysis, the train and test files are merged into one dataframe because the goal is **exploration and visualization**, not model evaluation.

## Project structure

```text
human-activity-pca/
|
|-- notebook/
|   `-- human_activity_pca_analysis.ipynb
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

## What each helper module does

- `src/data_loader.py`: loads feature names, activity labels, and merges the train/test splits.
- `src/preprocessing.py`: handles initial checks, class distribution, feature standardization, and before-vs-after dimension comparisons.
- `src/pca_analysis.py`: fits PCA, computes explained variance, finds variance-retention thresholds, and builds grouped loading summaries for presentation-friendly PC interpretation.
- `src/visualization.py`: creates the class distribution plot, scree plot, cumulative variance plot, and 2D PCA scatter plot.
- `src/utils.py`: handles project paths, output folders, random seeds, and saving tables/figures.

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

If you already have the dataset somewhere else, copy the full `UCI HAR Dataset` folder into `data/raw/`.

### 5. Run the notebook

From the project root:

```powershell
jupyter lab
```

Then open:

`notebook/human_activity_pca_analysis.ipynb`

The notebook is written to run from top to bottom without manual code edits after the dataset is in place.

## Outputs generated

When the notebook runs, it saves:

### Figures

- `outputs/figures/class_distribution.png`
- `outputs/figures/explained_variance_scree.png`
- `outputs/figures/cumulative_explained_variance.png`
- `outputs/figures/pca_2d_scatter.png`

### Tables

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

## What the results are expected to show

In this project run:

- the notebook makes the dimension reduction concrete by moving from **10,299 x 561** to **10,299 x 2** for visualization,
- the first two principal components capture about **57%** of the total variance,
- about **65** principal components are needed to retain roughly **90%** of the variance,
- about **104** principal components are needed to retain roughly **95%** of the variance,
- PCA gives a clear split between **dynamic activities** and **stationary activities**,
- `LAYING` tends to separate more clearly than `SITTING` and `STANDING`,
- `SITTING` and `STANDING` overlap the most in the 2D view,
- the walking classes are related but still show useful structure,
- PC1 behaves mostly like a broad movement-intensity axis, while PC2 adds finer separation between related activity styles.

So the main message is not that PCA solves the classification problem by itself. The message is that PCA gives a strong **low-dimensional summary** of a very high-dimensional sensor dataset.

## Short takeaway

PCA works well here as a first exploration tool:

- it reduces 561 features to a small number of interpretable components,
- it makes the dataset easy to visualize,
- it reveals broad activity structure,
- but it does not capture every detail in only two components.

That balance makes it a good topic for a clean classroom presentation on dimension reduction.
