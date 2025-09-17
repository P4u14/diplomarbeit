import os
import pandas as pd

from matplotlib import pyplot as plt

from visualization.plotter import Plotter


class HeatmapPlotter(Plotter):
    def __init__(self, columns, directory='heatmap_plots', cmap='viridis'):
        self.columns = columns
        self.directory = directory
        self.cmap = cmap

    def plot(self, dfs, exp_names, output_dir):
        import numpy as np
        # Ensure DataFrame column names have no extra whitespace
        for df in dfs:
            df.columns = df.columns.str.strip()
        # Convert all metric columns to numeric, coercing errors to NaN
        for df in dfs:
            for col in self.columns:
                if col in df.columns:
                    df[col] = df[col].apply(pd.to_numeric, errors='coerce')
        # Collect file names and dataset labels from first DataFrame (exclude aggregate 'All Datasets')
        df0 = dfs[0]
        print("Debug: columns in first DataFrame:", df0.columns.tolist())
        if 'Dataset' not in df0.columns or 'File Name' not in df0.columns:
            raise ValueError("Columns 'Dataset' or 'File Name' not found in DataFrame.")
        mask = df0['Dataset'] != 'All Datasets'
        # determine unique datasets sorted alphabetically
        unique_datasets = sorted(df0.loc[mask, 'Dataset'].unique())
        # build file list grouped by dataset, sorted within each group
        file_names = []
        dataset_labels = []
        for d in unique_datasets:
            group_files = sorted(df0.loc[df0['Dataset'] == d, 'File Name'].tolist())
            for fn in group_files:
                file_names.append(fn)
                dataset_labels.append(d)
        # ensure output directory exists
        dir_path = os.path.join(output_dir, self.directory)
        os.makedirs(dir_path, exist_ok=True)
        # Plot heatmap for each metric
        for col in self.columns:
            # build matrix: rows=files, cols=experiments
            matrix = []
            for fname in file_names:
                row_vals = []
                for df in dfs:
                    if col not in df.columns or 'File Name' not in df.columns:
                        raise ValueError(f"Columns 'File Name' or '{col}' not found in DataFrame for experiment.")
                    # find row for this file name
                    row = df.loc[df['File Name'] == fname]
                    if row.empty:
                        raise ValueError(f"File '{fname}' not found in DataFrame for experiment.")
                    row_vals.append(row.iloc[0][col])
                matrix.append(row_vals)
            matrix = np.array(matrix)
            # produce heatmap with auto-scaled color range
            safe_col = col.replace(" ", "_")
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
            # white separators between experiment columns (wider)
            ax.set_xticks(np.arange(-.5, len(exp_names), 1), minor=True)
            ax.grid(which='minor', axis='x', color='w', linestyle='-', linewidth=10)
            # white separators between dataset group rows (wider)
            group_boundaries = []
            for d in unique_datasets[:-1]:
                idxs = [i for i, dl in enumerate(dataset_labels) if dl == d]
                group_boundaries.append(idxs[-1] + 0.5)
            ax.set_yticks(group_boundaries, minor=True)
            ax.grid(which='minor', axis='y', color='w', linestyle='-', linewidth=10)
            ax.tick_params(which='minor', bottom=False, left=False)
            plt.colorbar(im)
            plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
            # set y-axis ticks at group centers with dataset labels
            positions = [ (next(i for i, dl in enumerate(dataset_labels) if dl == d) +
                            max(i for i, dl in enumerate(dataset_labels) if dl == d) ) / 2
                            for d in unique_datasets ]
            plt.yticks(positions, unique_datasets)
            plt.title(f"{col} across experiments per file (auto-scaled)")
            plt.tight_layout()
            out_path = os.path.join(dir_path, f"{safe_col}_heatmap.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved heatmap for '{col}' to {out_path}")
