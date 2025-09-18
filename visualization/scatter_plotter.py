import os
from matplotlib import pyplot as plt

from visualization.iplotter import IPlotter


class ScatterPlotter(IPlotter):
    def __init__(self, directory='scatter_plots'):
        self.directory = directory

    def plot(self, metrics, dfs, exp_names, output_dir):
        metric_x = metrics[0]
        metric_y = metrics[1]
        # color-coded scatter per experiment using all per-file rows
        plt.figure()
        for df, name in zip(dfs, exp_names):
            if metric_x not in df.columns or metric_y not in df.columns:
                print(f"Warning: Columns '{metric_x}' or '{metric_y}' not found in DataFrame for experiment '{name}'. Skipping this experiment.")
                continue
            # extract all metric values per file
            x_vals = df[metric_x].values
            y_vals = df[metric_y].values
            plt.scatter(x_vals, y_vals, label=name, alpha=0.7)
        # finalize plot
        plt.xlabel(metric_x)
        plt.ylabel(metric_y)
        plt.title(f"{metric_x} vs {metric_y} per-file across experiments")
        # axes start at zero
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        # legend mapping colors to experiments below the plot
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=len(exp_names))
        plt.tight_layout()

        # ensure output directory exists
        dir_path = os.path.join(output_dir, self.directory)
        os.makedirs(dir_path, exist_ok=True)

        # sanitize metric names for filename
        safe_x = metric_x.replace(" ", "_")
        safe_y = metric_y.replace(" ", "_")
        out_path = os.path.join(dir_path, f"scatter_{safe_x}_vs_{safe_y}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved scatter plot '{metric_x} vs {metric_y}' to {out_path}")
