import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sanitize_filename(name: str) -> str:
    return name.replace(' ', '_').replace('/', '_')


def main():
    # Path to mean CSV files
    pattern = os.path.join('data', 'Validation_Results', '*_mean.csv')
    files = glob.glob(pattern)
    if not files:
        print('No mean CSV files found')
        return

    # Load all dataframes keyed by experiment name
    data = {}
    for fp in files:
        exp = os.path.basename(fp).replace('_mean.csv', '')
        df = pd.read_csv(fp)
        data[exp] = df

    # Determine datasets and metrics
    sample_df = next(iter(data.values()))
    datasets = sample_df['Dataset'].tolist()
    metrics = [c for c in sample_df.columns if c != 'Dataset']
    # Separate special metrics for combined plot
    paired_metrics = [
        'Mean Number of GT Segments',
        'Mean Number of Segmentation Segments'
    ]
    other_metrics = [m for m in metrics if m not in paired_metrics]
    metrics = other_metrics

    # Create output directory
    outdir = 'data/Validation_Results/plots'
    os.makedirs(outdir, exist_ok=True)

    # Generate bar charts
    for dataset in datasets:
        for metric in metrics:
            values = []
            exps = []
            for exp, df in data.items():
                # Get value for this dataset and metric
                row = df.loc[df['Dataset'] == dataset]
                if not row.empty:
                    val = row.iloc[0][metric]
                    values.append(val)
                    exps.append(exp)
            if not values:
                continue

            # Sortiere Experimente alphabetisch
            pairs = sorted(zip(exps, values), key=lambda x: x[0])
            exps, values = zip(*pairs)

            plt.figure()
            # Achse bei y=0 zeichnen
            plt.axhline(0, color='black', linewidth=0.8)
            plt.bar(exps, values)
            plt.title(f"{metric} for {dataset}")
            plt.xlabel('Experiment')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            fname = f"{sanitize_filename(dataset)}_{sanitize_filename(metric)}.png"
            plt.savefig(os.path.join(outdir, fname))
            plt.close()

    print(f"Plots saved in {outdir}")

    # Combined bar charts for GT vs Segmentation segments
    for dataset in datasets:
        exps = []
        vals_gt = []
        vals_seg = []
        for exp, df in data.items():
            row = df.loc[df['Dataset'] == dataset]
            if not row.empty:
                exps.append(exp)
                vals_gt.append(row.iloc[0][paired_metrics[0]])
                vals_seg.append(row.iloc[0][paired_metrics[1]])
        if not exps:
            continue
        # Sort experiments alphabetically
        combined = sorted(zip(exps, vals_gt, vals_seg), key=lambda x: x[0])
        exps, vals_gt, vals_seg = zip(*combined)

        x = np.arange(len(exps))
        width = 0.35
        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linewidth=0.8)
        ax.bar(x - width/2, vals_gt, width, label=paired_metrics[0])
        ax.bar(x + width/2, vals_seg, width, label=paired_metrics[1])
        ax.set_xticks(x)
        ax.set_xticklabels(exps, rotation=45, ha='right')
        ax.set_title(f"GT vs Segmentation Segments for {dataset}")
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Mean Number of Segments')
        ax.legend()
        plt.tight_layout()
        fname = f"{sanitize_filename(dataset)}_GT_vs_Segments.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    print(f"Combined segment plots saved in {outdir}")

    # Combined bar charts for dimple center deviations
    dimple_metrics = [
        'Mean Dimples Center Left Deviation',
        'Mean Dimples Center Right Deviation',
        'Mean Dimples Center Left Deviation Abs',
        'Mean Dimples Center Right Deviation Abs'
    ]
    for dataset in datasets:
        exps = []
        vals = {m: [] for m in dimple_metrics}
        for exp, df in data.items():
            row = df.loc[df['Dataset'] == dataset]
            if not row.empty:
                exps.append(exp)
                for m in dimple_metrics:
                    vals[m].append(row.iloc[0][m])
        if not exps:
            continue
        # Sort experiments alphabetically
        combined = sorted(zip(exps, *(vals[m] for m in dimple_metrics)), key=lambda x: x[0])
        sorted_exps = [c[0] for c in combined]
        sorted_vals = [list(c[i] for c in combined) for i in range(1, len(dimple_metrics)+1)]

        x = np.arange(len(sorted_exps))
        total_width = 0.8
        width = total_width / len(dimple_metrics)
        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linewidth=0.8)
        for i, m in enumerate(dimple_metrics):
            ax.bar(x - total_width/2 + width*i + width/2, sorted_vals[i], width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_exps, rotation=45, ha='right')
        ax.set_title(f"Dimple Center Deviations for {dataset}")
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Deviation')
        ax.legend()
        plt.tight_layout()
        fname = f"{sanitize_filename(dataset)}_dimple_deviation.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    print(f"Combined dimple deviation plots saved in {outdir}")


if __name__ == '__main__':
    main()
