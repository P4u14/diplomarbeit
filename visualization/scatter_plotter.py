"""
ScatterPlotter module for creating scatter plots of experiment metrics.

This module defines the ScatterPlotter class, which inherits from BasePlotter and provides functionality to plot scatter plots for two selected metrics across multiple experiments. The plot can optionally include a legend and is saved to a specified output directory.

Classes:
    ScatterPlotter: Plots scatter plots for two metrics across experiments.
"""
import warnings
import os

from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class ScatterPlotter(BasePlotter):
    """
    ScatterPlotter creates scatter plots for two metrics across multiple experiments.

    Args:
        experiments (list): List of experiment names.
        metrics (list): List of two metric names to plot (x and y axes).
        directory (str, optional): Subdirectory for saving plots. Defaults to 'scatter_plots'.
        show_legend (bool, optional): Whether to display a legend. Defaults to True.
    """
    def __init__(self, experiments, metrics, directory='scatter_plots', show_legend=True):
        super().__init__(experiments, metrics, directory)
        self.show_legend = show_legend

    def plot(self, data_frames, output_dir):
        """
        Plots a scatter plot for the two specified metrics across the provided experiments.

        Args:
            data_frames (list): List of pandas DataFrames, one per experiment, containing the metrics.
            output_dir (str): Directory where the plot will be saved.

        The function checks that exactly two metrics are provided. It creates a scatter plot for each experiment, optionally with a legend. The plot is saved as a PNG file in the specified directory.
        """
        # require exactly two metrics
        n_exp = len(self.experiments)
        if len(self.metrics) != 2 or n_exp == 0:
            warnings.warn('ScatterPlotter needs exactly 2 metrics and at least 1 experiment.')
            return
        metric_x, metric_y = self.metrics
        # square window and 0.2 grid lines for Dice/precision metrics
        precision_mode = any('precision' in m.lower() for m in [metric_x, metric_y])
        if precision_mode:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            width = min(6, max(3, int(n_exp * 0.5)))
            fig, ax = plt.subplots(figsize=(width, 6))
        # ensure output folder exists
        os.makedirs(os.path.join(output_dir, self.directory), exist_ok=True)
        # scatter per experiment (uniform color if no legend)
        color = 'C0' if not self.show_legend else None
        for df, name in zip(data_frames, self.experiments):
            if metric_x not in df.columns or metric_y not in df.columns:
                warnings.warn(f"Columns '{metric_x}' or '{metric_y}' not found in experiment '{name}'. Skipping.")
                continue
            x_vals = df[metric_x].values
            y_vals = df[metric_y].values
            if self.show_legend:
                ax.scatter(x_vals, y_vals, label=name, alpha=0.7)
            else:
                ax.scatter(x_vals, y_vals, color=color, alpha=0.7)
        # labels and title
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(f"{metric_x} vs {metric_y} per-file across experiments")
        # axes start at zero
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        # square axes and grid lines for Dice/precision metrics
        if precision_mode:
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([round(i*0.2, 2) for i in range(6)])
            ax.set_yticks([round(i*0.2, 2) for i in range(6)])
        # optionally show legend below plot
        if self.show_legend:
            ncol = min(n_exp, 3)
            ax.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fontsize='small', frameon=False)
        # grid lines at each major tick
        ax.xaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        # layout adjustments
        fig.tight_layout()
        # increase bottom margin if legend shown
        if self.show_legend:
            fig.subplots_adjust(bottom=0.3)
        # save
        # construct and sanitize filename to avoid path separators
        filename = f"scatter_{metric_x.replace(' ', '_')}_vs_{metric_y.replace(' ', '_')}.png"
        filename = filename.replace('/', '_')
        self.save_plot(fig, output_dir, filename=filename)
