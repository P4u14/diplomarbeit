"""
Visualizer for generating plots from segmentation experiment results.

This module provides the Visualizer class, which coordinates the loading of experiment result CSVs
and the generation of plots using various plotter classes. It supports different plot types and
handles the correct file suffixes for each plotter.

Usage:
    visualizer = Visualizer(base_validation_path, plotters, output_dir)
    visualizer.visualize()

Classes:
    Visualizer: Loads experiment results and generates plots using the provided plotters.

Methods:
    - visualize: Main entry point. Loads CSVs for each experiment and plotter, then calls plotter.plot().
"""

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
    """
    Visualizer for generating plots from segmentation experiment results.
    Loads experiment result CSVs and generates plots using the provided plotter classes.

    Args:
        base_validation_path (str): Path to the directory containing experiment result CSVs.
        plotters (list): List of plotter instances to use for visualization.
        output_dir (str): Directory to save the generated plots.
    """
    def __init__(self, base_validation_path, plotters, output_dir):
        """
        Initialize the Visualizer.
        Args:
            base_validation_path (str): Path to the directory containing experiment result CSVs.
            plotters (list): List of plotter instances to use for visualization.
            output_dir (str): Directory to save the generated plots.
        """
        self.base_validation_path = base_validation_path
        self.plotters = plotters
        self.output_dir = output_dir

    def visualize(self):
        """
        Generate plots for all experiments and plotters.
        Loads the appropriate CSV files for each plotter and calls its plot() method.
        Creates the output directory if it does not exist.
        """
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