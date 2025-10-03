import warnings
import pandas as pd
from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class BubblePlotter(BasePlotter):
    """Bubble plot: size of bubble ∝ frequency of (x,y) pairs across all data."""
    def __init__(self, experiments, metrics, directory='bubble_plots'):
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
        # require exactly two metrics
        if len(self.metrics) != 2 or not data_frames:
            warnings.warn('BubblePlotter needs exactly 2 metrics and at least one dataset.')
            return
        metric_x, metric_y = self.metrics
        # concatenate all experiments' data
        df = pd.concat(data_frames, ignore_index=True)
        if metric_x not in df.columns or metric_y not in df.columns:
            warnings.warn(f"Columns '{metric_x}' or '{metric_y}' not found.")
            return
        # drop NaNs
        df = df.dropna(subset=[metric_x, metric_y])
        # group and count occurrences of each pair
        counts_df = df.groupby([metric_x, metric_y]).size().reset_index(name='count')
        x_vals = counts_df[metric_x].values
        y_vals = counts_df[metric_y].values
        sizes = counts_df['count'].values * 20  # scale factor for bubble size
        # plot
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(x_vals, y_vals, s=sizes, alpha=0.6)
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(f"Bubble plot of {metric_x} vs {metric_y} (size ∝ frequency)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        # save
        filename = f"bubble_{metric_x.replace(' ', '_')}_vs_{metric_y.replace(' ', '_')}.png"
        self.save_plot(fig, output_dir, filename=filename)

