import warnings

import numpy as np
from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class LinePlotter(BasePlotter):
    def __init__(self, experiments, metrics, directory='line_plots'):
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
        # number of experiments and metrics
        n_exp = len(self.experiments)
        n_met = len(self.metrics)
        if n_exp == 0 or n_met == 0:
            warnings.warn('No experiments or metrics to plot.')
            return

       # numeric x positions for experiments
        x = np.arange(len(self.experiments))

        # derive group labels from first DataFrame
        df0 = data_frames[0]
        all_labels = list(df0['Dataset'].unique())

        # include All Datasets and dataset names, exclude only status labels
        dataset_groups = [d for d in all_labels if d not in ('Healthy', 'Sick')]

        # include All Datasets, Healthy, Sick if present
        status_groups = [g for g in ('All Datasets', 'Healthy', 'Sick') if g in all_labels]

        # iterate metrics
        for metric in self.metrics:
            # prepare and save by-dataset plot
            fig = self.create_by_dataset_plots(data_frames, dataset_groups, metric, n_exp, x)
            filename = f"{metric.replace(' ','_')}_by_dataset_line.png"
            self.save_plot(fig, output_dir, filename=filename)

            # prepare and save by-health-status plot
            fig = self.create_bay_health_status_plots(data_frames, metric, n_exp, status_groups, x)
            filename = f"{metric.replace(' ','_')}_by_status_line.png"
            self.save_plot(fig, output_dir, filename=filename)

    def create_by_dataset_plots(self, data_frames, dataset_groups, metric, n_exp, x):
        fig, ax = plt.subplots(figsize=(max(3, int((n_exp + 1) * 0.3)), 6))
        for grp in dataset_groups:
            y = []
            for df in data_frames:
                if metric in df.columns and grp in df['Dataset'].values:
                    val = df.loc[df['Dataset'] == grp, metric].iloc[0]
                else:
                    val = np.nan
                y.append(val)
            ax.plot(x, y, marker='o', label=grp, linestyle='dashed')
        ax.set_title(f"{metric} by dataset")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(self.experiments, rotation=45, ha='right')
        ax.legend()
        # draw horizontal grid lines at each y-tick for better readability
        ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        # if ratio metric, add horizontal grid lines every 0.1
        if metric.lower() in ['dice', 'precision', 'recall']:
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.0001, 0.1))
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        return fig

    def create_bay_health_status_plots(self, data_frames, metric, n_exp, status_groups, x):
        fig, ax = plt.subplots(figsize=(max(3, int((n_exp + 1) * 0.3)), 6))
        for grp in status_groups:
            y = []
            for df in data_frames:
                if metric in df.columns and grp in df['Dataset'].values:
                    val = df.loc[df['Dataset'] == grp, metric].iloc[0]
                else:
                    val = np.nan
                y.append(val)
            ax.plot(x, y, marker='o', label=grp, linestyle='dashed')
        ax.set_title(f"{metric} by health status")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(self.experiments, rotation=45, ha='right')
        ax.legend()
        # draw horizontal grid lines at each y-tick for better readability
        ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        # if ratio metric, add horizontal grid lines every 0.1
        if metric.lower() in ['dice', 'precision', 'recall']:
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.0001, 0.1))
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        return fig
