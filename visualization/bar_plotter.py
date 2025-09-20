import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.base_plotter import BasePlotter


class BarPlotter(BasePlotter):
    def __init__(self, experiments, metrics, directory='bar_charts'):
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
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
            ax.yaxis.set_major_formatter(FuncFormatter(self._format_sec))
            # add horizontal grid lines at each major tick for readability
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.tight_layout()

    def format_x_axis(self, ax, bar_width, total_width, x):
        ax.set_xticks(x + total_width / 2 - bar_width / 2)
        ax.set_xticklabels(self.experiments, rotation=45, ha='right')

    def create_bar_per_metric(self, ax, bar_width, data_frames, x):
        for i, metric in enumerate(self.metrics):
            vals = []
            is_duration = 'duration' in metric.lower()
            for df in data_frames:
                if metric not in df.columns:
                    warnings.warn(f"Column '{metric}' not found in DataFrame, using NaN.")
                    vals.append(np.nan)
                else:
                    # select aggregate row
                    row = df.loc[df['Dataset'] == 'All Datasets']
                    if row.empty:
                        vals.append(np.nan)
                    else:
                        raw = row.iloc[0][metric]
                        val = self._parse_time(raw) if is_duration else raw
                        vals.append(val)
            ax.bar(x + i * bar_width, vals, width=bar_width, label=metric)

    @staticmethod
    def _parse_time(ts: str) -> float:
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
    def _format_sec(x, pos):
        h = int(x) // 3600
        m = (int(x) % 3600) // 60
        s = int(x) % 60
        return f"{h:02}:{m:02}:{s:02}"
