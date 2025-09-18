import os
from pandas import read_csv

from visualization.bar_plotter import BarPlotter
from visualization.box_plotter import BoxPlotter
from visualization.line_plotter import LinePlotter
from visualization.heatmap_plotter import HeatmapPlotter

# --- Configuration: define base path, experiment names and output dir here ---
BASE_VALIDATION_PATH = 'data/Results/Validation'
EXPERIMENTS = [
    # Experiment identifiers (without suffix)
    'Atlas_Experiment01',
    'Atlas_Experiment02',
    'Atlas_Experiment03',
]
COLUMNS = [
    # Column names to plot
    # 'Dice'
    'Total duration',
    'Average duration per image'
]
OUTPUT_DIR = 'data/Results/Plots/Atlas_Experiments'
# Define plotter instances to use
PLOTTERS = [
    BarPlotter(COLUMNS),
    BoxPlotter(COLUMNS),
    LinePlotter(COLUMNS),
    HeatmapPlotter(COLUMNS)
]
# ---------------------------------------------------------------


class Visualizer:
    def __init__(self, csv_paths, plotters, output_dir):
        self.experiment_names = csv_paths
        self.plotters = plotters
        self.output_dir = output_dir

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Plot each plotter: build file paths from base path and experiment names
        for plotter in self.plotters:
            # choose suffix based on plotter type
            if isinstance(plotter, (BoxPlotter, HeatmapPlotter)):
                suffix = '_all.csv'
            else:
                suffix = '_mean.csv'
            # assemble file paths
            paths = [os.path.join(BASE_VALIDATION_PATH, name + suffix) for name in self.experiment_names]
            dfs = [read_csv(p) for p in paths]
            # use experiment_names as labels
            plotter.plot(dfs, self.experiment_names, self.output_dir)


def main():
    # Instantiate and run Visualizer with configuration above
    viz = Visualizer(EXPERIMENTS, PLOTTERS, OUTPUT_DIR)
    viz.run()


if __name__ == '__main__':
    main()
