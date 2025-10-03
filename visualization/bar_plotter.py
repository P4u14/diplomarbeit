import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.base_plotter import BasePlotter


class BarPlotter(BasePlotter):
    def __init__(self, experiments, metrics, directory='bar_charts', highlighted_experiments=None, experiment_labels=None, show_ms_in_duration=False):
        super().__init__(experiments, metrics, directory)
        self.highlighted_experiments = highlighted_experiments or []
        self.experiment_labels = experiment_labels or {}
        self.show_ms_in_duration = show_ms_in_duration

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
        # y-axis limits und formatting
        # if any metric is a ratio, assume [0,1]
        if any(m.lower() in ['dice', 'precision', 'recall'] for m in self.metrics):
            ax.set_ylim(0, 1)
            # add horizontal grid lines at every 0.1 for easier reading
            ax.set_yticks(np.arange(0, 1.0001, 0.1))
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        # duration formatter und grid lines
        if any('duration' in m.lower() for m in self.metrics):
            # Ermittle das Maximum der Balken für duration
            all_bars = [b for c in ax.containers for b in c]
            max_val = max([b.get_height() for b in all_bars], default=0)
            ylim = max(1.05, max_val * 1.05)  # Mindestens 1 Sekunde, etwas Puffer
            ax.set_ylim(0, ylim)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: self._format_sec(x, pos, self.show_ms_in_duration)))
            # add horizontal grid lines at each major tick for readability
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.tight_layout()

    def format_x_axis(self, ax, bar_width, total_width, x):
        ax.set_xticks(x + total_width / 2 - bar_width / 2)
        ax.set_xticklabels(self.experiments, rotation=45, ha='right')

    def create_bar_per_metric(self, ax, bar_width, data_frames, x):
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
            # Für die Legende: Ein Balken pro Experiment with Label
            for idx, exp in enumerate(self.experiments):
                label = self.experiment_labels.get(exp, exp)
                if label not in bar_labels and exp in self.highlighted_experiments:
                    bar_handles.append(bars[idx])
                    bar_labels.append(label)
        # Legende nur, wenn mindestens ein Experiment hervorgehoben ist
        if highlight_active:
            ax.legend(bar_handles, bar_labels)

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
    def _format_sec(x, pos, show_ms=False):
        h = int(x) // 3600
        m = (int(x) % 3600) // 60
        s = int(x) % 60
        ms = int(round((x - int(x)) * 1000))
        if show_ms:
            return f"{h:02}:{m:02}:{s:02}.{ms:03}"
        else:
            return f"{h:02}:{m:02}:{s:02}"
