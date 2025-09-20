import warnings

from matplotlib import pyplot as plt

from visualization.base_plotter import BasePlotter


class BoxPlotter(BasePlotter):
    def __init__(self, experiments, metrics, directory='box_plots'):
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
        # validate inputs
        n_exp = len(self.experiments)
        n_met = len(self.metrics)
        if n_exp == 0 or n_met == 0:
            warnings.warn('No experiments or metrics to plot.')
            return
        # dynamic figure width
        width = max(3, int((n_exp + 1) * 0.3))
        # iterate metrics
        for metric in self.metrics:
            # skip if missing column
            if any(metric not in df.columns for df in data_frames):
                warnings.warn(f"Column '{metric}' not found in one of the DataFrames. Skipping box plot for this column.")
                continue
            # collect data arrays
            data = [df[metric].dropna().values for df in data_frames]
            # create plot
            fig, ax = plt.subplots(figsize=(width, 6))
            ax.boxplot(data)
            ax.set_title(f"{metric} distribution across experiments")
            ax.set_ylabel(metric)
            ax.set_xticklabels(self.experiments, rotation=45, ha='right')
            # grid lines at each y-tick
            ax.yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
            ax.set_ylim(bottom=0)
            # save plot
            filename = f"{metric.replace(' ', '_')}_box_plot.png"
            self.save_plot(fig, output_dir, filename=filename)
