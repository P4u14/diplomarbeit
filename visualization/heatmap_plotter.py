import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class HeatmapPlotter(BasePlotter):
    def __init__(self, experiments, metrics, directory='heatmap_plots', cmap='viridis'):
        super().__init__(experiments, metrics, directory)
        self.cmap = cmap

    def plot(self, data_frames, output_dir):
        # Ensure DataFrame column names have no extra whitespace
        for df in data_frames:
            df.columns = df.columns.str.strip()

        # Convert all metric columns to numeric, coercing errors to NaN
        for df in data_frames:
            for metric in self.metrics:
                if metric in df.columns:
                    df[metric] = df[metric].apply(pd.to_numeric, errors='coerce')

        # Collect file names and dataset labels from first DataFrame (exclude aggregate 'All Datasets')
        df0 = data_frames[0]
        if 'Dataset' not in df0.columns or 'File Name' not in df0.columns:
            raise ValueError("Columns 'Dataset' or 'File Name' not found in DataFrame.")

        # determine valid entries excluding aggregate
        mask = df0['Dataset'] != 'All Datasets'

        # determine unique datasets sorted alphabetically
        unique_datasets = sorted(df0.loc[mask, 'Dataset'].unique())

        # build file list grouped by dataset and health status
        file_names = []
        dataset_labels = []
        status_labels = []

        # convert Sick column to numeric
        df0['Sick'] = pd.to_numeric(df0['Sick'], errors='coerce')
        for d in unique_datasets:
            df_d = df0[mask & (df0['Dataset'] == d)]
            # healthy if Sick is 0 or -1
            healthy = sorted(df_d.loc[df_d['Sick'].isin([0, -1]), 'File Name'].tolist())
            sick = sorted(df_d.loc[~df_d['Sick'].isin([0, -1]), 'File Name'].tolist())
            for fn in healthy:
                file_names.append(fn); dataset_labels.append(d); status_labels.append('Healthy')
            for fn in sick:
                file_names.append(fn); dataset_labels.append(d); status_labels.append('Sick')

        # ensure output directory exists
        dir_path = os.path.join(output_dir, self.directory)
        os.makedirs(dir_path, exist_ok=True)

        # Plot heatmap for each metric
        for metric in self.metrics:
            # skip if column missing in any DataFrame
            if any(metric not in df.columns for df in data_frames):
                print(f"Warning: Column '{metric}' not found in one of the DataFrames. Skipping heatmap for this column.")
                continue
            # build matrix: rows=files, cols=experiments
            matrix = []
            for filename in file_names:
                row_vals = []
                for df in data_frames:
                    # assume File Name exists (checked earlier); col existence checked above
                    row = df.loc[df['File Name'] == filename]
                    if row.empty:
                        raise ValueError(f"File '{filename}' not found in DataFrame for experiment.")
                    row_vals.append(row.iloc[0][metric])
                matrix.append(row_vals)
            matrix = np.array(matrix)

            # produce heatmap with auto-scaled color range
            # sanitize metric name for filename (replace spaces and slashes)
            safe_col = metric.replace(' ', '_').replace('/', '_')

            # create figure and axis
            fig, ax = plt.subplots(figsize=(max(6, int(len(self.experiments)*0.3)), 6))

            # mask missing values and set mask color to black
            m = np.ma.masked_invalid(matrix)
            cmap = plt.get_cmap(self.cmap).copy()
            cmap.set_bad(color='black')

            # auto-scale color to valid data range
            vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
            im = ax.imshow(m, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
            # draw separators
            # set major ticks for experiment labels
            ax.set_xticks(range(len(self.experiments)))
            # draw white vertical separators between experiments
            for i in range(len(self.experiments) - 1):
                ax.axvline(i + 0.5, color='w', linewidth=2)
            # compute separators and y-ticks grouping by health
            status_boundaries = []
            dataset_boundaries = []
            positions = []
            labels = []
            for d in unique_datasets:
                # indices for this dataset block
                idxs = [i for i, dl in enumerate(dataset_labels) if dl == d]
                healthy_idxs = [i for i in idxs if status_labels[i] == 'Healthy']
                sick_idxs = [i for i in idxs if status_labels[i] == 'Sick']
                # separator between healthy and sick rows
                if healthy_idxs and sick_idxs:
                    status_boundaries.append(healthy_idxs[-1] + 0.5)
                # separator after entire dataset block
                dataset_boundaries.append(idxs[-1] + 0.5)
                # tick label positions and labels
                if healthy_idxs:
                    positions.append((healthy_idxs[0] + healthy_idxs[-1]) / 2)
                    labels.append(f"{d} Healthy")
                if sick_idxs:
                    positions.append((sick_idxs[0] + sick_idxs[-1]) / 2)
                    labels.append(f"{d} Sick")
            # draw horizontal separators: health status and dataset boundaries
            for b in status_boundaries:
                ax.axhline(b, color='w', linewidth=7)
            for b in dataset_boundaries:
                ax.axhline(b, color='w', linewidth=10)
            fig.colorbar(im)
            ax.set_xticklabels(self.experiments, rotation=45, ha='right')
            # set y-axis ticks for status subgroups
            plt.yticks(positions, labels)
            plt.title(f"{metric} across experiments per file")
            fig.tight_layout()
            # save via BasePlotter
            filename = f"{safe_col}_heatmap.png"
            self.save_plot(fig, output_dir, filename=filename)
