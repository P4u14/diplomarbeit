Example command to run downstream semantic segmentation training

```bash
python -m ssl.downstream.train_segmentation --config ssl/downstream/config.yaml
```

## Command Line Options
| Option   | Type | Default Value   | Description                    |
|----------|------|-----------------|--------------------------------|
| --config | str  | (required path) | Path to the configuration file |