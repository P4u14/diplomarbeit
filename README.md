# diplomarbeit

This repository contains code and data management for a thesis project on medical image segmentation, evaluation, and visualization. It provides scripts and modules for preprocessing, segmentation, evaluation, and result visualization, as well as DVC-based data versioning.

### DVC initialization

```bash
dvc init
```

### DVC add data to be tracked (folder or files)

```bash
dvc add data/Atlas_Data
```
- Note: the data folder will contain all the data sets for this project. Nevertheless, it is best practice to add each data set separately to DVC. In this way, you can track changes to each data set individually.

### DVC push data to (local) remote storage
- DVC supports various remote storage options, including local directories, S3 buckets, GCS buckets, etc.
- TU Dresden's Nextcloud can be connected but 10 GB storage limit is not enough for this project.
- In this example, we will use a local directory as the remote storage.
- Make sure to create the directory first if it doesn't exist.

```bash
dvc remote add -d localremote /Users/paula/Documents/DA/dvcstore
dvc push
```

### Track changes to the data set in git

```bash
git add data/Atlas_Data.dvc .dvc/config .dvcignore
git commit -m "Track Atlas_Data with DVC"
```

## Entry Points

The following scripts serve as entry points for the main project workflows:

- `segment.py` – Segmentation of images
- `evaluate.py` – Evaluation of segmentation results
- `visualize.py` – Visualization and plotting of results

> **Note:** Paths to data, results, and configuration files, as well as variable names, must be adjusted directly in the respective scripts to match your local setup and requirements.

## Project Structure

- `preprocessing/` – Preprocessing scripts for image and ROI extraction
- `segmenter/` – Segmentation models and experiment runners
- `evaluation/` – Evaluation metrics and experiment evaluation
- `postprocessing/` – Postprocessing and refinement scripts
- `visualization/` – Plotting and visualization tools
- `data/` – Data folder (tracked by DVC)
- `README.md` – Project documentation
