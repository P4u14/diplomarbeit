import os

from pandas import read_csv

from visualization.box_plotter import BoxPlotter
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.scatter_plotter import ScatterPlotter


class Visualizer:
    def __init__(self, base_validation_path, csv_paths, plotters, metrics, output_dir):
        self.base_validation_path = base_validation_path
        self.experiment_names = csv_paths
        self.plotters = plotters
        self.metrics = metrics
        self.output_dir = output_dir

    def visualize(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for plotter in self.plotters:
            if isinstance(plotter, (BoxPlotter, HeatmapPlotter, ScatterPlotter)):
                suffix = '_all.csv'
            else:
                suffix = '_mean.csv'
            paths = [os.path.join(self.base_validation_path, name + suffix) for name in self.experiment_names]
            dfs = [read_csv(p) for p in paths]
            # use experiment_names as labels
            plotter.plot(self.metrics, dfs, self.experiment_names, self.output_dir)