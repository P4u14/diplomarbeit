import os
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.plotter import Plotter


def _parse_time(ts: str) -> float:
    parts = ts.split(':')
    h = int(parts[0]); m = int(parts[1])
    if '.' in parts[2]:
        s_str, ms_str = parts[2].split('.')
        s = int(s_str); ms = int(ms_str)
    else:
        s = int(parts[2]); ms = 0
    return h*3600 + m*60 + s + ms/1000

def _format_sec(x, pos):
    h = int(x) // 3600
    m = (int(x) % 3600) // 60
    s = int(x) % 60
    ms = int((x - int(x)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

class BarPlotter(Plotter):
    def __init__(self, columns, directory='bar_charts'):
        self.columns = columns
        self.directory = directory

    def plot(self, dfs, exp_names, output_dir):
        for col in self.columns:
            is_duration = 'duration' in col.lower()
            # skip if column missing in any DataFrame
            if any(col not in df.columns for df in dfs):
                print(f"Warning: Column '{col}' not found in one of the DataFrames. Skipping Bar chart for this column.")
                continue
            values = []
            for df in dfs:
                # col existence already checked
                # Select value from row where 'Dataset' column equals 'All Datasets'
                if 'Dataset' in df.columns and 'All Datasets' in df['Dataset'].values:
                    row = df.loc[df['Dataset'] == 'All Datasets']
                    raw = row.iloc[0][col]
                    if is_duration:
                        val = _parse_time(raw)
                    else:
                        val = raw
                    values.append(val)
            plt.figure()
            plt.bar(exp_names, values)
            # if plotting duration, format y-axis ticks
            if is_duration:
                ax = plt.gca()
                ax.yaxis.set_major_formatter(FuncFormatter(_format_sec))
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