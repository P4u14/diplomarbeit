import warnings
import numpy as np
from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class AvgDiceByNSegmentsPlotter(BasePlotter):
    """
    Plot average of one metric (e.g., Dice) as a function of another (e.g., N Segments GT) per experiment.
    """
    def __init__(self, experiments, metrics, directory='avg_dice_by_nsegments'):  # metrics: [x_metric, y_metric]
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
        if len(self.metrics) != 2:
            warnings.warn('AvgDiceByNSegmentsPlotter needs exactly 2 metrics: [x_metric, y_metric].')
            return
        metric_x, metric_y = self.metrics
        if not data_frames:
            warnings.warn('No data to plot.')
            return
        # collect all unique x values across experiments
        x_vals_all = set()
        for df in data_frames:
            if metric_x in df.columns:
                x_vals_all.update(df[metric_x].dropna().unique())
        if not x_vals_all:
            warnings.warn(f"Metric '{metric_x}' not found in any data.")
            return
        sorted_x = sorted(x_vals_all)
        # prepare plot
        fig, ax = plt.subplots(figsize=(max(3, int((len(sorted_x)+1)*0.3)), 6))
        # plot per experiment
        for df, name in zip(data_frames, self.experiments):
            if metric_x not in df.columns or metric_y not in df.columns:
                warnings.warn(f"Skipping experiment '{name}': missing '{metric_x}' or '{metric_y}'.")
                continue
            grouped = df.groupby(metric_x)[metric_y].mean()
            y_vals = [grouped.get(x, np.nan) for x in sorted_x]
            ax.plot(sorted_x, y_vals, marker='o', label=name)
        ax.set_title(f"Average {metric_y} by {metric_x} per experiment")
        ax.set_xlabel(metric_x)
        ax.set_ylabel(f"Average {metric_y}")
        ax.set_xticks(sorted_x)
        # place legend below plot with multiple columns like scatter plotter
        ncol = min(len(self.experiments), 3)
        ax.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', frameon=False)
        ax.xaxis.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        # filename
        fn = f"avg_{metric_y.replace(' ', '_')}_by_{metric_x.replace(' ', '_')}.png"
        self.save_plot(fig, output_dir, filename=fn)
