import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# User configuration: specify experiment CSV paths and metrics to plot
EXPERIMENT_FILES = [
    'data/Results/Validation/Atlas_Experiment01_mean.csv',
    'data/Results/Validation/Atlas_Experiment02_mean.csv',
    'data/Results/Validation/Atlas_Experiment03_mean.csv',
    'data/Results/Validation/Atlas_Experiment04_mean.csv',
    'data/Results/Validation/Atlas_Experiment05_mean.csv',
    'data/Results/Validation/Atlas_Experiment06_mean.csv',
    'data/Results/Validation/Atlas_Experiment07_mean.csv',
    'data/Results/Validation/Atlas_Experiment08_mean.csv',
    'data/Results/Validation/Atlas_Experiment09_mean.csv',
    'data/Results/Validation/Atlas_Experiment10_mean.csv',
    'data/Results/Validation/Atlas_Experiment11_mean.csv',
    'data/Results/Validation/Atlas_Experiment12_mean.csv',
    'data/Results/Validation/Atlas_Experiment13_mean.csv',
    'data/Results/Validation/Atlas_Experiment14_mean.csv',
    'data/Results/Validation/Atlas_Experiment15_mean.csv',
    'data/Results/Validation/Atlas_Experiment16_mean.csv',
    'data/Results/Validation/Atlas_Experiment17_mean.csv',
    'data/Results/Validation/Atlas_Experiment18_mean.csv',
    'data/Results/Validation/Atlas_Experiment19_mean.csv',
    'data/Results/Validation/Atlas_Experiment20_mean.csv',
    'data/Results/Validation/Atlas_Experiment21_mean.csv',
]

METRICS = [
    'Mean Dice',
    'Mean Precision',
    'Mean Recall',
]

METRIC_GROUPS = [
    ['Mean N GT Segments', 'Mean N Pred Segments']
]

OUTPUT_DIR = 'data/Results/Plots'
MAX_XTICKS = 10  # maximum number of x-axis labels to show


def sanitize_filename(name: str) -> str:
    return name.replace(' ', '_').replace('/', '_')


def main():
    # Load files from user-defined list
    files = EXPERIMENT_FILES
    if not files:
        print('No experiment files provided')
        return

    # Load all dataframes keyed by experiment name
    data = {}
    for fp in files:
        exp = os.path.basename(fp).replace('_mean.csv', '')
        df = pd.read_csv(fp)
        data[exp] = df

    # Determine datasets
    sample_df = next(iter(data.values()))
    datasets = sample_df['Dataset'].tolist()
    # Use user-defined metrics and ensure existence
    metrics = [m for m in METRICS if m in sample_df.columns]
    missing = [m for m in METRICS if m not in sample_df.columns]
    if missing:
        print(f"Warning: The following metrics are not found in data and will be skipped: {missing}")

    # Compute global y-limits for individual metrics
    metric_limits = {}
    for metric in metrics:
        vals = []
        for df in data.values():
            vals.extend(df[metric].dropna().tolist())
        if vals:
            metric_limits[metric] = (min(vals), max(vals))
    # Compute global y-limits for metric groups
    group_limits = {}
    for group in METRIC_GROUPS:
        valid = [m for m in group if m in sample_df.columns]
        if not valid:
            continue
        vals = []
        for df in data.values():
            for m in valid:
                vals.extend(df[m].dropna().tolist())
        if vals:
            group_limits[tuple(valid)] = (min(vals), max(vals))

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            # apply consistent y-axis scale
            if metric in metric_limits:
                plt.ylim(metric_limits[metric])
            plt.title(f"{metric} for {dataset}")
            plt.xlabel('Experiment')
            plt.ylabel(metric)
            # skip xticks if too many experiments
            n = len(exps)
            if n > MAX_XTICKS:
                step = int(np.ceil(n / MAX_XTICKS))
                ticks = list(range(0, n, step))
                plt.xticks(ticks, [exps[i] for i in ticks], rotation=45, ha='right')
            else:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            fname = f"{sanitize_filename(dataset)}_{sanitize_filename(metric)}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()

    print(f"Individual plots saved in {OUTPUT_DIR}")

    # Combined plots for metric groups
    for dataset in datasets:
        for group in METRIC_GROUPS:
            # ensure metrics exist
            valid = [m for m in group if m in sample_df.columns]
            if not valid:
                continue
            exps = []
            vals = {m: [] for m in valid}
            for exp, df in data.items():
                row = df.loc[df['Dataset'] == dataset]
                if not row.empty:
                    exps.append(exp)
                    for m in valid:
                        vals[m].append(row.iloc[0][m])
            if not exps:
                continue
            # Sort experiments alphabetically
            combined = sorted(zip(exps, *(vals[m] for m in valid)), key=lambda x: x[0])
            exps_sorted = [c[0] for c in combined]
            sorted_vals = [ [c[i] for c in combined] for i in range(1, len(valid)+1) ]
            x = np.arange(len(exps_sorted))
            total_width = 0.8
            width = total_width / len(valid)
            plt.figure()
            plt.axhline(0, color='black', linewidth=0.8)
            for i, m in enumerate(valid):
                plt.bar(x - total_width/2 + width*i + width/2, sorted_vals[i], width, label=m)
            # apply consistent y-axis for this metric group
            key = tuple(valid)
            if key in group_limits:
                plt.ylim(group_limits[key])
            # skip xticks if too many experiments
            n = len(exps_sorted)
            if n > MAX_XTICKS:
                step = int(np.ceil(n / MAX_XTICKS))
                ticks = x[::step]
                labels = [exps_sorted[i] for i in range(0, n, step)]
                plt.xticks(ticks, labels, rotation=45, ha='right')
            else:
                plt.xticks(x, exps_sorted, rotation=45, ha='right')
            plt.title(f"{' & '.join(valid)} for {dataset}")
            plt.xlabel('Experiment')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            fname = f"{sanitize_filename(dataset)}_{sanitize_filename('_'.join(valid))}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()
    print(f"Combined group plots saved in {OUTPUT_DIR}")

    # end of plotting
    # Generate line plots for each metric across all experiments and subgroups
    for metric in metrics:
        exps = sorted(data.keys())
        plt.figure()
        for dataset in datasets:
            vals = []
            for exp in exps:
                df = data[exp]
                row = df.loc[df['Dataset'] == dataset]
                if not row.empty:
                    vals.append(row.iloc[0][metric])
                else:
                    vals.append(np.nan)
            # plot line with markers, skip if all values are NaN
            if not all(pd.isna(v) for v in vals):
                plt.plot(exps, vals, marker='o', label=dataset)
        # apply consistent y-axis scale
        if metric in metric_limits:
            plt.ylim(metric_limits[metric])
        plt.title(f"{metric} across experiments and subgroups")
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        # skip xticks if too many experiments
        n = len(exps)
        if n > MAX_XTICKS:
            step = int(np.ceil(n / MAX_XTICKS))
            ticks = list(range(0, n, step))
            plt.xticks(ticks, [exps[i] for i in ticks], rotation=45, ha='right')
        else:
            plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{sanitize_filename(metric)}_across_experiments.png"))
        plt.close()
    print(f"Line plots for metrics across experiments saved in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
