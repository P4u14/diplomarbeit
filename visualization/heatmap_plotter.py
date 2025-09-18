import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from visualization.iplotter import IPlotter


class HeatmapPlotter(IPlotter):
    def __init__(self, directory='heatmap_plots', cmap='viridis'):
        self.directory = directory
        self.cmap = cmap

    def plot(self, metrics, dfs, exp_names, output_dir):
        # Ensure DataFrame column names have no extra whitespace
        for df in dfs:
            df.columns = df.columns.str.strip()
        # Convert all metric columns to numeric, coercing errors to NaN
        for df in dfs:
            for metric in metrics:
                if metric in df.columns:
                    df[metric] = df[metric].apply(pd.to_numeric, errors='coerce')
        # Collect file names and dataset labels from first DataFrame (exclude aggregate 'All Datasets')
        df0 = dfs[0]
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
        for metric in metrics:
            # skip if column missing in any DataFrame
            if any(metric not in df.columns for df in dfs):
                print(f"Warning: Column '{metric}' not found in one of the DataFrames. Skipping heatmap for this column.")
                continue
            # build matrix: rows=files, cols=experiments
            matrix = []
            for fname in file_names:
                row_vals = []
                for df in dfs:
                    # assume File Name exists (checked earlier); col existence checked above
                    row = df.loc[df['File Name'] == fname]
                    if row.empty:
                        raise ValueError(f"File '{fname}' not found in DataFrame for experiment.")
                    row_vals.append(row.iloc[0][metric])
                matrix.append(row_vals)
            matrix = np.array(matrix)
            # produce heatmap with auto-scaled color range
            safe_col = metric.replace(" ", "_")
            plt.figure()
            # mask missing values and set mask color to black
            m = np.ma.masked_invalid(matrix)
            cmap = plt.get_cmap(self.cmap).copy()
            cmap.set_bad(color='black')
            # auto-scale color to valid data range
            auto_vmin = np.nanmin(matrix)
            auto_vmax = np.nanmax(matrix)
            im = plt.imshow(m, cmap=cmap, aspect='auto', interpolation='nearest', vmin=auto_vmin, vmax=auto_vmax)
            ax = plt.gca()
            # set major ticks for experiment labels
            ax.set_xticks(range(len(exp_names)))
            # draw white vertical separators between experiments
            for i in range(len(exp_names) - 1):
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
            plt.colorbar(im)
            plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
            # set y-axis ticks for status subgroups
            plt.yticks(positions, labels)
            plt.title(f"{metric} across experiments per file (auto-scaled)")
            plt.tight_layout()
            out_path = os.path.join(dir_path, f"{safe_col}_heatmap.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved heatmap for '{metric}' to {out_path}")
