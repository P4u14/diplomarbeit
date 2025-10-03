"""
bar_plotter.py - Bar plotter for experiment metrics.

This module defines the BarPlotter class, which creates bar charts for selected metrics across multiple experiments. The bar charts visualize metric values for each experiment and can highlight specific experiments. Plots are saved to the specified output directory.

Classes:
    BarPlotter: Plots bar charts for metrics across experiments, with optional highlighting and custom labels.
"""
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.base_plotter import BasePlotter


class BarPlotter(BasePlotter):
    """
    BarPlotter creates bar charts for selected metrics across multiple experiments.

    Args:
        experiments (list): List of experiment names.
        metrics (list): List of metric names to plot as bar charts.
        directory (str, optional): Subdirectory for saving plots. Defaults to 'bar_charts'.
        highlighted_experiments (list, optional): List of experiment names to highlight. Defaults to None.
        experiment_labels (dict, optional): Mapping of experiment names to display labels. Defaults to None.
        show_ms_in_duration (bool, optional): Whether to show milliseconds in duration formatting. Defaults to False.
    """
    def __init__(self, experiments, metrics, directory='bar_charts', highlighted_experiments=None, experiment_labels=None, show_ms_in_duration=False):
        super().__init__(experiments, metrics, directory)
        self.highlighted_experiments = highlighted_experiments or []
        self.experiment_labels = experiment_labels or {}
        self.show_ms_in_duration = show_ms_in_duration

    def plot(self, data_frames, output_dir):
        """
        Plots bar charts for each specified metric across the provided experiments.

        The bar charts visualize metric values for each experiment. Optionally, specific experiments can be highlighted and custom labels can be used. The plot is saved as a PNG file in the specified directory.

        Args:
            data_frames (list): List of pandas DataFrames, one per experiment, containing the metrics.
            output_dir (str): Directory where the plots will be saved.
        """
        # number of experiments and metrics
        n_exp = len(self.experiments)
        n_met = len(self.metrics)
        if n_exp == 0 or n_met == 0:
            warnings.warn('No experiments or metrics to plot.')
            return

        # positions for grouped bars
        x = np.arange(n_exp)
        total_width = 0.5
        bar_width = total_width / n_met

        # prepare figure
        fig, ax = plt.subplots(figsize=(max(3, int((n_exp + 1) * 0.3)), 6))

        # plot each metric's bars
        self.create_bar_per_metric(ax, bar_width, data_frames, x)

        # formatting
        self.format_x_axis(ax, bar_width, total_width, x)
        self.format_y_axis(ax)

        # save plot
        filename = '_'.join([m.replace(' ', '_') for m in self.metrics]) + '_bar_chart.png'
        self.save_plot(fig, output_dir, filename=filename)

    def format_y_axis(self, ax):
        """
        Format the y-axis for the bar chart, including label, legend, limits, and grid lines.

        Args:
            ax (matplotlib.axes.Axes): The axes to format.
        """
        # if only one metric, use it as y-axis label; otherwise show legend
        if len(self.metrics) == 1:
            ax.set_ylabel(self.metrics[0])
        else:
            ax.legend()
        # y-axis limits and formatting
        # if any metric is a ratio, assume [0,1]
        if any(m.lower() in ['dice', 'precision', 'recall'] for m in self.metrics):
            ax.set_ylim(0, 1)
            # add horizontal grid lines at every 0.1 for easier reading
            ax.set_yticks(np.arange(0, 1.0001, 0.1))
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        # duration formatter and grid lines
        if any('duration' in m.lower() for m in self.metrics):
            # Determine the maximum bar height for duration
            all_bars = [b for c in ax.containers for b in c]
            max_val = max([b.get_height() for b in all_bars], default=0)
            ylim = max(1.05, max_val * 1.05)  # At least 1 second, with some padding
            ax.set_ylim(0, ylim)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: self._format_sec(x, pos, self.show_ms_in_duration)))
            # add horizontal grid lines at each major tick for readability
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.tight_layout()

    def format_x_axis(self, ax, bar_width, total_width, x):
        """
        Format the x-axis for the bar chart, including tick positions and labels.

        Args:
            ax (matplotlib.axes.Axes): The axes to format.
            bar_width (float): The width of each bar.
            total_width (float): The total width allocated for all bars in a group.
            x (np.ndarray): The x positions for the groups.
        """
        ax.set_xticks(x + total_width / 2 - bar_width / 2)
        ax.set_xticklabels(self.experiments, rotation=45, ha='right')

    def create_bar_per_metric(self, ax, bar_width, data_frames, x):
        """
        Create and plot bars for each metric and experiment, with optional highlighting.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            bar_width (float): The width of each bar.
            data_frames (list): List of pandas DataFrames, one per experiment.
            x (np.ndarray): The x positions for the groups.
        """
        default_color = 'C0'
        highlight_color = 'green'
        bar_handles = []
        bar_labels = []
        highlight_active = bool(self.highlighted_experiments)
        for i, metric in enumerate(self.metrics):
            vals = []
            colors = []
            for idx, exp in enumerate(self.experiments):
                df = data_frames[idx]
                if metric not in df.columns:
                    warnings.warn(f"Column '{metric}' not found in DataFrame, using NaN.")
                    vals.append(np.nan)
                else:
                    row = df.loc[df['Dataset'] == 'All Datasets']
                    if row.empty:
                        vals.append(np.nan)
                    else:
                        raw = row.iloc[0][metric]
                        val = self._parse_time(raw) if 'duration' in metric.lower() else raw
                        vals.append(val)
                if exp in self.highlighted_experiments:
                    colors.append(highlight_color)
                else:
                    colors.append(default_color)
            bars = ax.bar(x + i * bar_width, vals, width=bar_width, color=colors)
            # For the legend: one bar per experiment with label
            for idx, exp in enumerate(self.experiments):
                label = self.experiment_labels.get(exp, exp)
                if label not in bar_labels and exp in self.highlighted_experiments:
                    bar_handles.append(bars[idx])
                    bar_labels.append(label)
        # Show legend only if at least one experiment is highlighted
        if highlight_active:
            ax.legend(bar_handles, bar_labels)

    @staticmethod
    def _parse_time(ts: str) -> float:
        """
        Parse a time string in the format HH:MM:SS or HH:MM:SS.mmm to seconds as float.

        Args:
            ts (str): Time string to parse.

        Returns:
            float: Time in seconds.
        """
        parts = ts.split(':')
        h = int(parts[0])
        m = int(parts[1])
        if '.' in parts[2]:
            s_str, ms_str = parts[2].split('.')
            s = int(s_str)
            ms = int(ms_str)
        else:
            s = int(parts[2])
            ms = 0
        return h * 3600 + m * 60 + s + ms / 1000

    @staticmethod
    def _format_sec(x, pos, show_ms=False):
        """
        Format a float number of seconds as a time string HH:MM:SS or HH:MM:SS.mmm.

        Args:
            x (float): Time in seconds.
            pos: Unused, required for matplotlib formatter.
            show_ms (bool, optional): Whether to show milliseconds. Defaults to False.

        Returns:
            str: Formatted time string.
        """
        h = int(x) // 3600
        m = (int(x) % 3600) // 60
        s = int(x) % 60
        ms = int(round((x - int(x)) * 1000))
        if show_ms:
            return f"{h:02}:{m:02}:{s:02}.{ms:03}"
        else:
            return f"{h:02}:{m:02}:{s:02}"
