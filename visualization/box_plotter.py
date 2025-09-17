import os

from matplotlib import pyplot as plt

from visualization.plotter import Plotter


class BoxPlotter(Plotter):
    def __init__(self, columns, directory='box_plots'):
        self.columns = columns
        self.directory = directory

    def plot(self, dfs, exp_names, output_dir):
        for col in self.columns:
            data = []
            for df in dfs:
                if col in df.columns:
                    data.append(df[col].dropna().values)
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame for experiment.")
            plt.figure()
            plt.boxplot(data, labels=exp_names)
            plt.title(f"{col} distribution across experiments")
            plt.ylabel(col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.ylim(bottom=0)
            # ensure output subdirectory exists
            dir_path = os.path.join(output_dir, self.directory)
            os.makedirs(dir_path, exist_ok=True)
            # sanitize column name for filename
            safe_col = col.replace(" ", "_")
            out_path = os.path.join(dir_path, f"{safe_col}_box_plot.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved box plot for '{col}' to {out_path}")