import os
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.iplotter import IPlotter


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
    return f"{h:02}:{m:02}:{s:02}"

class BarPlotter(IPlotter):
    def __init__(self, directory='bar_charts'):
        self.directory = directory

    def plot(self, metrics, dfs, exp_names, output_dir):
        for metric in metrics:
            is_duration = 'duration' in metric.lower()
            # skip if column missing in any DataFrame
            if any(metric not in df.columns for df in dfs):
                print(f"Warning: Column '{metric}' not found in one of the DataFrames. Skipping Bar chart for this column.")
                continue
            values = []
            for df in dfs:
                # metric existence already checked
                # Select value from row where 'Dataset' column equals 'All Datasets'
                if 'Dataset' in df.columns and 'All Datasets' in df['Dataset'].values:
                    row = df.loc[df['Dataset'] == 'All Datasets']
                    raw = row.iloc[0][metric]
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
            plt.title(f"Mean {metric} across experiments")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(bottom=0)
            plt.tight_layout()
            # Create bar_charts directory if it doesn't exist
            dir_path = os.path.join(output_dir, self.directory)
            os.makedirs(dir_path, exist_ok=True)
            # Replace spaces in column name for filename
            safe_col = metric.replace(" ", "_")
            out_path = os.path.join(dir_path, f"{safe_col}_bar_chart.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved bar chart for '{metric}' to {out_path}")