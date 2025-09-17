import os

from matplotlib import pyplot as plt

from visualization.plotter import Plotter


class BarPlotter(Plotter):
    def __init__(self, columns, directory='bar_charts'):
        self.columns = columns
        self.directory = directory

    def plot(self, dfs, exp_names, output_dir):
        for col in self.columns:
            values = []
            for df in dfs:
                if col in df.columns:
                    # Select value from row where 'Dataset' column equals 'All Datasets'
                    if 'Dataset' in df.columns and 'All Datasets' in df['Dataset'].values:
                        row = df.loc[df['Dataset'] == 'All Datasets']
                        values.append(row.iloc[0][col])
                    else:
                        raise ValueError("No row with 'Dataset' == 'All Datasets' found in DataFrame.")
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame for experiment.")
            plt.figure()
            plt.bar(exp_names, values)
            plt.title(f"Mean {col} across experiments")
            plt.ylabel(col)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(bottom=0)
            plt.tight_layout()
            # Create bar_charts directory if it doesn't exist
            dir_path = os.path.join(output_dir, self.directory)
            os.makedirs(dir_path, exist_ok=True)
            # Replace spaces in column name for filename
            safe_col = col.replace(" ", "_")
            out_path = os.path.join(dir_path, f"{safe_col}_bar_chart.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved bar chart for '{col}' to {out_path}")