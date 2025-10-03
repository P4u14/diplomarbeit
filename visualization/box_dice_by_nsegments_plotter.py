import warnings
import numpy as np
from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class BoxDiceByNSegmentsPlotter(BasePlotter):
    """
    Boxplot of one metric (e.g., Dice) grouped by another (e.g., N Segments GT),
    combining data across multiple experiments.
    """
    def __init__(self, experiments, metrics, directory='box_dice_by_nsegments'):
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
        # require exactly two metrics
        if len(self.metrics) != 2:
            warnings.warn('BoxDiceByNSegmentsPlotter needs exactly 2 metrics: [x_metric, y_metric].')
            return
        metric_x, metric_y = self.metrics
        # combine data across experiments
        all_data = []
        for df in data_frames:
            if metric_x in df.columns and metric_y in df.columns:
                sub = df[[metric_x, metric_y]].dropna()
                all_data.append(sub)
        if not all_data:
            warnings.warn('No data to plot.')
            return
        df_all = np.concatenate([d.values for d in all_data], axis=0)
        # df_all is array of shape (N, 2)
        x_vals = df_all[:, 0]
        y_vals = df_all[:, 1]
        # unique sorted x values
        x_unique = np.unique(x_vals)
        # collect y per x group
        data = [y_vals[x_vals == xv] for xv in x_unique]
        # create boxplot
        fig, ax = plt.subplots(figsize=(max(3, int((len(x_unique)+1)*0.3)), 6))
        ax.boxplot(data, positions=x_unique, widths=0.6)
        ax.set_title(f"Distribution of {metric_y} by {metric_x}")
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_xticks(x_unique)
        ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in x_unique], rotation=45, ha='right')
        ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
        plt.tight_layout()
        # save plot
        filename = f"box_{metric_y.replace(' ', '_')}_by_{metric_x.replace(' ', '_')}.png"
        self.save_plot(fig, output_dir, filename=filename)

