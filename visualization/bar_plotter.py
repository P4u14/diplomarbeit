import os
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.plotter import Plotter


class BarPlotter(Plotter):
    def __init__(self, columns, directory='bar_charts'):
        self.columns = columns
        self.directory = directory

    def plot(self, dfs, exp_names, output_dir):
        for col in self.columns:
            is_duration = 'duration' in col.lower()
            if is_duration:
                # helper to convert h:mm:ss.ms to seconds
                def parse_time(ts: str) -> float:
                    parts = ts.split(':')
                    h = int(parts[0]); m = int(parts[1])
                    if '.' in parts[2]:
                        s_str, ms_str = parts[2].split('.')
                        s = int(s_str); ms = int(ms_str)
                    else:
                        s = int(parts[2]); ms = 0
                    return h*3600 + m*60 + s + ms/1000
                # formatter to display seconds back to h:mm:ss.ms
                def format_sec(x, pos):
                    h = int(x) // 3600
                    m = (int(x) % 3600) // 60
                    s = int(x) % 60
                    ms = int((x - int(x)) * 1000)
                    return f"{h:02}:{m:02}:{s:02}.{ms:03}"
            values = []
            for df in dfs:
                if col in df.columns:
                    # Select value from row where 'Dataset' column equals 'All Datasets'
                    if 'Dataset' in df.columns and 'All Datasets' in df['Dataset'].values:
                        row = df.loc[df['Dataset'] == 'All Datasets']
                        raw = row.iloc[0][col]
                        if is_duration:
                            val = parse_time(raw)
                        else:
                            val = raw
                        values.append(val)
                    else:
                        raise ValueError("No row with 'Dataset' == 'All Datasets' found in DataFrame.")
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame for experiment.")
            plt.figure()
            plt.bar(exp_names, values)
            # if plotting duration, format y-axis ticks
            if is_duration:
                ax = plt.gca()
                ax.yaxis.set_major_formatter(FuncFormatter(format_sec))
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