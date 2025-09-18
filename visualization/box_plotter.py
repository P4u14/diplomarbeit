import os

from matplotlib import pyplot as plt

from visualization.iplotter import IPlotter


class BoxPlotter(IPlotter):
    def __init__(self, directory='box_plots'):
        self.directory = directory

    def plot(self, metrics, dfs, exp_names, output_dir):
        for metric in metrics:
            # skip if column missing in any DataFrame
            if any(metric not in df.columns for df in dfs):
                print(f"Warning: Column '{metric}' not found in one of the DataFrames. Skipping box plot for this column.")
                continue
            data = []
            for df in dfs:
                data.append(df[metric].dropna().values)
            plt.figure()
            plt.boxplot(data, labels=exp_names)
            plt.title(f"{metric} distribution across experiments")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.ylim(bottom=0)
            # ensure output subdirectory exists
            dir_path = os.path.join(output_dir, self.directory)
            os.makedirs(dir_path, exist_ok=True)
            # sanitize column name for filename
            safe_col = metric.replace(" ", "_")
            out_path = os.path.join(dir_path, f"{safe_col}_box_plot.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved box plot for '{metric}' to {out_path}")