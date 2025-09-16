import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap

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
    'data/Results/Validation/Atlas_Experiment22_mean.csv',
    'data/Results/Validation/Atlas_Experiment23_mean.csv',
    'data/Results/Validation/Atlas_Experiment24_mean.csv',
    'data/Results/Validation/Atlas_Experiment25_mean.csv',
    'data/Results/Validation/Atlas_Experiment26_mean.csv',
    'data/Results/Validation/Atlas_Experiment27_mean.csv',
    'data/Results/Validation/Atlas_Experiment28_mean.csv',
    'data/Results/Validation/Atlas_Experiment29_mean.csv',
    'data/Results/Validation/Atlas_Experiment30_mean.csv',
    'data/Results/Validation/Atlas_Experiment31_mean.csv',
    'data/Results/Validation/Atlas_Experiment32_mean.csv',
    'data/Results/Validation/Atlas_Experiment33_mean.csv',
    'data/Results/Validation/Atlas_Experiment34_mean.csv',
    'data/Results/Validation/Atlas_Experiment35_mean.csv',
    'data/Results/Validation/Atlas_Experiment36_mean.csv',
    'data/Results/Validation/Atlas_Experiment42_mean.csv',
    'data/Results/Validation/Atlas_Experiment43_mean.csv',
    'data/Results/Validation/Atlas_Experiment44_mean.csv',
    'data/Results/Validation/Atlas_Experiment45_mean.csv',
    'data/Results/Validation/Atlas_Experiment46_mean.csv',
    'data/Results/Validation/Atlas_Experiment47_mean.csv',
    'data/Results/Validation/Atlas_Experiment48_mean.csv',
    'data/Results/Validation/Atlas_Experiment49_mean.csv',
    'data/Results/Validation/Atlas_Experiment50_mean.csv',
    'data/Results/Validation/Atlas_Experiment51_mean.csv',
    'data/Results/Validation/Atlas_Experiment52_mean.csv',
    'data/Results/Validation/Atlas_Experiment53_mean.csv',
    'data/Results/Validation/Atlas_Experiment54_mean.csv',
    'data/Results/Validation/Atlas_Experiment55_mean.csv',
    'data/Results/Validation/Atlas_Experiment56_mean.csv',
    'data/Results/Validation/Atlas_Experiment57_mean.csv',
    'data/Results/Validation/Atlas_Experiment58_mean.csv',
    'data/Results/Validation/Atlas_Experiment59_mean.csv',
    'data/Results/Validation/Atlas_Experiment60_mean.csv',
    'data/Results/Validation/Atlas_Experiment61_mean.csv',
    'data/Results/Validation/Atlas_Experiment62_mean.csv',
    'data/Results/Validation/Atlas_Experiment63_mean.csv',
    'data/Results/Validation/Atlas_Experiment64_mean.csv',
    'data/Results/Validation/Atlas_Experiment65_mean.csv',
    'data/Results/Validation/Atlas_Experiment66_mean.csv',
    'data/Results/Validation/Atlas_Experiment67_mean.csv',
    'data/Results/Validation/Atlas_Experiment68_mean.csv',
    'data/Results/Validation/Atlas_Experiment69_mean.csv',
    'data/Results/Validation/Atlas_Experiment70_mean.csv',
    'data/Results/Validation/Atlas_Experiment71_mean.csv',
    'data/Results/Validation/Atlas_Experiment72_mean.csv',
    'data/Results/Validation/Atlas_Experiment73_mean.csv',
]

METRICS = [
    'Mean Dice',
    'Mean Precision',
    'Mean Recall',
    'Mean Center Pred Success'
]

METRIC_GROUPS = [
    ['Mean N GT Segments', 'Mean N Pred Segments']
]

OUTPUT_DIR = 'data/Results/Plots'
MAX_XTICKS = 17  # maximum number of x-axis labels to show
# Title wrapping configuration
TITLE_WRAP_WIDTH = 40  # characters per line before wrap
def wrap_title(s: str) -> str:
    return "\n".join(textwrap.wrap(s, TITLE_WRAP_WIDTH))
# User configuration: specify which subgroup sets to plot per metric across experiments
SUBGROUP_GROUPS = [
    # ['*'],  # All subgroups
    ['Healthy', 'Sick'],
    ['gkge', 'wip', 'mBrace', 'skolioseKielce']
]

# User configuration: named experiment groups for boxplots
BOXPLOT_EXPERIMENT_GROUPS = [
    {
        'label': 'Atlas Data',
        'files': [
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
    'data/Results/Validation/Atlas_Experiment22_mean.csv',
    'data/Results/Validation/Atlas_Experiment23_mean.csv',
    'data/Results/Validation/Atlas_Experiment24_mean.csv',
    'data/Results/Validation/Atlas_Experiment25_mean.csv',
    'data/Results/Validation/Atlas_Experiment26_mean.csv',
    'data/Results/Validation/Atlas_Experiment27_mean.csv',
    'data/Results/Validation/Atlas_Experiment28_mean.csv',
    'data/Results/Validation/Atlas_Experiment29_mean.csv',
    'data/Results/Validation/Atlas_Experiment30_mean.csv',
    'data/Results/Validation/Atlas_Experiment31_mean.csv',
    'data/Results/Validation/Atlas_Experiment32_mean.csv',
    'data/Results/Validation/Atlas_Experiment33_mean.csv',
    'data/Results/Validation/Atlas_Experiment34_mean.csv',
    'data/Results/Validation/Atlas_Experiment35_mean.csv',
    'data/Results/Validation/Atlas_Experiment36_mean.csv',
        ]
    },
    {
        'label': 'Atlas Data BMI Percentile',
        'files': [
    'data/Results/Validation/Atlas_Experiment42_mean.csv',
    'data/Results/Validation/Atlas_Experiment43_mean.csv',
    'data/Results/Validation/Atlas_Experiment44_mean.csv',
    'data/Results/Validation/Atlas_Experiment45_mean.csv',
    'data/Results/Validation/Atlas_Experiment46_mean.csv',
    'data/Results/Validation/Atlas_Experiment47_mean.csv',
    'data/Results/Validation/Atlas_Experiment48_mean.csv',
    'data/Results/Validation/Atlas_Experiment49_mean.csv',
    'data/Results/Validation/Atlas_Experiment50_mean.csv',
    'data/Results/Validation/Atlas_Experiment51_mean.csv',
    'data/Results/Validation/Atlas_Experiment52_mean.csv',
    'data/Results/Validation/Atlas_Experiment53_mean.csv',
    'data/Results/Validation/Atlas_Experiment54_mean.csv',
    'data/Results/Validation/Atlas_Experiment55_mean.csv',
    'data/Results/Validation/Atlas_Experiment56_mean.csv',
    'data/Results/Validation/Atlas_Experiment57_mean.csv',
    'data/Results/Validation/Atlas_Experiment58_mean.csv',
    'data/Results/Validation/Atlas_Experiment59_mean.csv',
    'data/Results/Validation/Atlas_Experiment60_mean.csv',
    'data/Results/Validation/Atlas_Experiment61_mean.csv',
    'data/Results/Validation/Atlas_Experiment62_mean.csv',
    'data/Results/Validation/Atlas_Experiment63_mean.csv',
    'data/Results/Validation/Atlas_Experiment64_mean.csv',
    'data/Results/Validation/Atlas_Experiment65_mean.csv',
    'data/Results/Validation/Atlas_Experiment66_mean.csv',
    'data/Results/Validation/Atlas_Experiment67_mean.csv',
    'data/Results/Validation/Atlas_Experiment68_mean.csv',
    'data/Results/Validation/Atlas_Experiment69_mean.csv',
    'data/Results/Validation/Atlas_Experiment70_mean.csv',
    'data/Results/Validation/Atlas_Experiment71_mean.csv',
    'data/Results/Validation/Atlas_Experiment72_mean.csv',
    'data/Results/Validation/Atlas_Experiment73_mean.csv',
        ]
    },
    # {'label': 'Group A', 'files': ['path/to/Exp1_mean.csv', 'path/to/Exp2_mean.csv']},
]


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
            # apply consistent y-axis scale with lower bound 0
            if metric in metric_limits:
                _, ymax = metric_limits[metric]
                plt.ylim(0, ymax)
            else:
                plt.ylim(bottom=0)
            # add horizontal grid lines with dynamic spacing
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            step = 0.1 if y_max <= 1 else 0.5
            yt = np.arange(y_min, y_max + step, step)
            ax.set_yticks(yt)
            ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)
            plt.title(wrap_title(f"{metric} for {dataset}"))
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
            # Sort experiments alphabetisch
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
            # apply consistent y-axis scale with lower bound 0 for this metric group
            key = tuple(valid)
            if key in group_limits:
                _, ymax = group_limits[key]
                plt.ylim(0, ymax)
            else:
                plt.ylim(bottom=0)
            # add horizontal grid lines with dynamic spacing
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            step = 0.1 if y_max <= 1 else 0.5
            yt = np.arange(y_min, y_max + step, step)
            ax.set_yticks(yt)
            ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)
            # skip xticks if too many experiments
            n = len(exps_sorted)
            if n > MAX_XTICKS:
                step = int(np.ceil(n / MAX_XTICKS))
                ticks = x[::step]
                labels = [exps_sorted[i] for i in range(0, n, step)]
                plt.xticks(ticks, labels, rotation=45, ha='right')
            else:
                plt.xticks(x, exps_sorted, rotation=45, ha='right')
            plt.title(wrap_title(f"{' & '.join(valid)} for {dataset}"))
            plt.xlabel('Experiment')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            fname = f"{sanitize_filename(dataset)}_{sanitize_filename('_'.join(valid))}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()
    print(f"Combined group plots saved in {OUTPUT_DIR}")

    # end of plotting
    # Generate line plots for each metric across all experiments for user-defined subgroup sets
    exps = sorted(data.keys())
    for metric in metrics:
        for group in SUBGROUP_GROUPS:
            # expand '*' to mean all subgroups
            if '*' in group:
                valid_subs = list(datasets)
            else:
                valid_subs = [s for s in group if s in datasets]
            if not valid_subs:
                continue
            # prepare values per subgroup to determine grid step
            vals_dict = {}
            for ds in valid_subs:
                lst = []
                for exp in exps:
                    row = data[exp].loc[data[exp]['Dataset'] == ds]
                    lst.append(row.iloc[0][metric] if not row.empty else np.nan)
                vals_dict[ds] = lst
            plt.figure()
            for ds, lst in vals_dict.items():
                if not all(pd.isna(v) for v in lst):
                    plt.plot(exps, lst, marker='o', label=ds)
            # decide grid spacing: use .5 if 'All' subgroup max >1, else .1
            grid_step = 0.1
            if 'All' in vals_dict:
                valid_all = [v for v in vals_dict['All'] if not pd.isna(v)]
                if valid_all and max(valid_all) > 1:
                    grid_step = 0.1
            # apply consistent y-axis scale with lower bound 0
            if metric in metric_limits:
                _, ymax = metric_limits[metric]
                plt.ylim(0, ymax)
            else:
                plt.ylim(bottom=0)
            # add horizontal grid lines with dynamic spacing based on y-axis maximum
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            step = 0.1 if y_max <= 1 else 0.5
            yt = np.arange(y_min, y_max + step, step)
            ax.set_yticks(yt)
            ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)
            title = f"{metric}: {' & '.join(valid_subs)} across experiments"
            plt.title(wrap_title(title))
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
            fname = f"{sanitize_filename(metric)}_{sanitize_filename('_'.join(valid_subs))}_across.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()
    print(f"Line plots for metrics and subgroup sets saved in {OUTPUT_DIR}")

    # Generate boxplots: one figure per metric with box for 'All' and each category
    for metric in metrics:
        # collect values per category and overall
        data_per_cat = {ds: [] for ds in datasets}
        all_vals = []
        for exp, df in data.items():
            for ds in datasets:
                row = df.loc[df['Dataset'] == ds]
                if not row.empty:
                    val = row.iloc[0][metric]
                    data_per_cat[ds].append(val)
                    all_vals.append(val)
        # prepare boxplot data
        cats = ['All'] + datasets
        vals = [all_vals] + [data_per_cat[ds] for ds in datasets]
        plt.figure()
        ax = plt.gca()
        ax.boxplot(vals)
        ax.set_xticklabels(cats, rotation=45, ha='right')
        # set title and wrap
        ax.set_title(wrap_title(f"{metric} Boxplot: All & per category"))
        # y-axis starts at 0 and uses consistent max from metric_limits
        if metric in metric_limits:
            _, ymax = metric_limits[metric]
        else:
            ymax = max(all_vals) if all_vals else 0
        ax.set_ylim(0, ymax)
        # grid lines dynamic spacing
        step = 0.1 if ymax <= 1 else 0.5
        yt = np.arange(0, ymax + step, step)
        ax.set_yticks(yt)
        ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        fname = f"{sanitize_filename(metric)}_boxplot.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close()
    print(f"Boxplots saved in {OUTPUT_DIR}")
    # Generate experiment-group boxplots per metric
    # ensure custom boxplot directory exists
    custom_dir = os.path.join(OUTPUT_DIR, 'custom_boxplots')
    os.makedirs(custom_dir, exist_ok=True)
    for metric in metrics:
        group_vals = []
        group_labels = []
        for grp in BOXPLOT_EXPERIMENT_GROUPS:
            files = grp.get('files', [])
            label = grp.get('label') or os.path.basename(files[0]).replace('_mean.csv','') if files else ''
            vals = []
            for fp in files:
                exp = os.path.basename(fp).replace('_mean.csv', '')
                if exp in data:
                    vals.extend(data[exp][metric].dropna().tolist())
            if vals:
                group_vals.append(vals)
                group_labels.append(label)
        if group_vals:
            plt.figure()
            ax = plt.gca()
            ax.boxplot(group_vals)
            ax.set_xticklabels(group_labels, rotation=45, ha='right')
            ax.set_title(wrap_title(f"{metric} Boxplot: experiment groups"))
            # y-axis from 0 to global max
            ymax = metric_limits.get(metric, (0, max(max(v) for v in group_vals)))[1]
            ax.set_ylim(0, ymax)
            # grid lines dynamic
            step = 0.1 if ymax <= 1 else 0.5
            yt = np.arange(0, ymax + step, step)
            ax.set_yticks(yt)
            ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            fname = f"{sanitize_filename(metric)}_exp_group_boxplot.png"
            plt.savefig(os.path.join(custom_dir, fname))
            plt.close()
    print(f"Experiment-group boxplots saved in {custom_dir}")


if __name__ == '__main__':
    main()
