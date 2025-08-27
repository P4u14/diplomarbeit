Example command to run jigsaw permutation pretext task

```bash
python -m ssl.pretext.jigsaw_absolute_position.JigsawAbsPosTrain data/ML_Pretext_All_Filtered --checkpoint data/SSL_Pretext/JigsawAbsPos/Experiment01 --gpu 0 --batch 64 --classes 9 --epochs 30 --pp torso square
```

## Command Line Options

| Option               | Type  | Default Value               | Description                                                 |
|----------------------|-------|-----------------------------|-------------------------------------------------------------|
| data                 | str   | (required path)             | Path to the Imagenet folder (training data)                 |
| --model              | str   | None                        | Path to a pretrained model                                  |
| --classes            | int   | 9                           | Number of patches (must be a square number)                 |
| --gpu                | int   | 0                           | GPU ID (0 = first GPU, -1 = CPU/MPS)                        |
| --epochs             | int   | 70                          | Number of training epochs                                   |
| --iter_start         | int   | 0                           | Starting value for iteration counter                        |
| --batch              | int   | 64                          | Batch size                                                  |
| --checkpoint         | str   | data/SSL_Pretext/JigsawPerm | Folder for checkpoints/models                               |
| --lr                 | float | 0.001                       | Learning rate for the SGD optimizer                         |
| --cores              | int   | 0                           | Number of CPU cores for data loading                        |
| -e, --evaluate       | Flag  | False                       | Only evaluate on the validation dataset, no training        |
| --pp                 | list  | ['square']                  | Preprocessing pipeline in order. Example: --pp torso square |
| --torso-target-ratio | float | 0.714 (5/7)                 | TorsoRoiPreprocessor.target_ratio (e.g. 5/7)                |
| --square-resize      | int   | 256                         | SquareImagePreprocessor: Resize edge (Default 256)          |
| --square-crop        | int   | 255                         | SquareImagePreprocessor: CenterCrop edge (Default 255)      |
