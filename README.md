# diplomarbeit

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
- TU Dresdens Nextcloud can be connected but 10 GB storage limit is not enough for this project.
- In this example, we will use a local directory as the remote storage.
- Make sure to create the directory first if it doesn't exist.

```bash
dvc remote add -d localremote /Users/paula/Documents/DA/dvcstore
dvc push
```