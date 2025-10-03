import os

from pandas import read_csv

from visualization.box_plotter import BoxPlotter
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.scatter_plotter import ScatterPlotter
from visualization.metric_count_bar_plotter import MetricCountBarPlotter
from visualization.bubble_plotter import BubblePlotter
from visualization.avg_dice_by_nsegments_plotter import AvgDiceByNSegmentsPlotter
from visualization.box_dice_by_nsegments_plotter import BoxDiceByNSegmentsPlotter


class Visualizer:
    def __init__(self, base_validation_path, plotters, output_dir):
        self.base_validation_path = base_validation_path
        self.plotters = plotters
        self.output_dir = output_dir

    def visualize(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for plotter in self.plotters:
            if isinstance(plotter, (BoxPlotter, HeatmapPlotter, ScatterPlotter, MetricCountBarPlotter, BubblePlotter, AvgDiceByNSegmentsPlotter, BoxDiceByNSegmentsPlotter)):
                suffix = '_all.csv'
            else:
                suffix = '_mean.csv'
            paths = [os.path.join(self.base_validation_path, name + suffix) for name in plotter.experiments]
            dfs = [read_csv(str(p)) for p in paths]
            # use experiment_names as labels
            plotter.plot(dfs, self.output_dir)