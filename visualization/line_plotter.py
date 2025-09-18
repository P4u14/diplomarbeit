import os

from matplotlib import pyplot as plt

from visualization.iplotter import IPlotter


class LinePlotter(IPlotter):
    def __init__(self, directory='line_plots'):
        self.directory = directory

    def plot(self, metrics, dfs, exp_names, output_dir):
        for metric in metrics:
            # skip if column missing in any DataFrame
            if any(metric not in df.columns for df in dfs):
                print(f"Warning: Column '{metric}' not found in one of the DataFrames. Skipping line plot for this column.")
                continue
            # Define groups of Dataset categories to plot
            groups = {
                'datasets': ['wip', 'mBrace', 'gkge', 'skolioseKielce', 'All Datasets'],
                'health': ['Sick', 'Healthy', 'All Datasets'],
            }
            # ensure output subdirectory exists
            dir_path = os.path.join(output_dir, self.directory)
            os.makedirs(dir_path, exist_ok=True)
            # sanitize column name for filenames
            safe_col = metric.replace(" ", "_")
            # Generate line plot for each group
            for group_name, items in groups.items():
                plt.figure()
                for item in items:
                    series = []
                    for df in dfs:
                        if 'Dataset' not in df.columns or item not in df['Dataset'].values:
                            raise ValueError(f"No row with 'Dataset' == '{item}' found in DataFrame.")
                        row = df.loc[df['Dataset'] == item]
                        series.append(row.iloc[0][metric])
                    plt.plot(exp_names, series, marker='o', label=item)
                plt.title(f"Mean {metric} trend across experiments by {group_name}")
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.ylim(bottom=0)
                plt.tight_layout()
                safe_group = group_name.replace(" ", "_")
                out_path = os.path.join(dir_path, f"{safe_col}_{safe_group}_line_plot.png")
                plt.savefig(out_path)
                plt.close()
                print(f"Saved {group_name} line plot for '{metric}' to {out_path}")
