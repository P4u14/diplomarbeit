Example command to select permutations for jigsaw permutation pretext task

```bash
python ssl/pretext/jigsaw_permutation/permutations/select_permutations.py --classes 100
```

## Command Line Options

| Option         | Type   | Default | Description                                                                 |
|---------------|--------|---------|-----------------------------------------------------------------------------|
| --classes     | int    | 1000    | Number of permutations to select                                            |
| --selection   | str    | max     | Sample selection per iteration based on hamming distance: [max], [mean]     |


