"""
base_plotter.py - Abstract base class for experiment plotters.

This module defines the BasePlotter class, which provides common functionality for all plotter classes in the visualization package. It implements the IPlotter interface and provides a standard method for saving plots.

Classes:
    BasePlotter: Abstract base class for plotters, providing experiment/metric management and plot saving.
"""
import os

from matplotlib import pyplot as plt

from visualization.iplotter import IPlotter


class BasePlotter(IPlotter):
    """
    Abstract base class for plotter classes.

    Provides common initialization and a standard plot saving method for all plotters.

    Args:
        experiments (list): List of experiment names.
        metrics (list): List of metric names to plot.
        directory (str): Subdirectory for saving plots.
    """
    def __init__(self, experiments, metrics, directory):
        self.experiments = experiments
        self.metrics = metrics
        self.directory = directory

    def plot(self, data_frames, output_dir):
        """
        Abstract method to generate plots from the provided dataframes and save them to the specified output directory.

        Args:
            data_frames (list): List of pandas DataFrames containing the data to plot.
            output_dir (str): Directory where the generated plots should be saved.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the plot method.")

    def save_plot(self, fig, output_dir, filename=None):
        """
        Save a matplotlib figure in the configured directory, using an optional filename.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            output_dir (str): The base output directory.
            filename (str, optional): The filename for the saved plot. If None, a name is generated from the metrics.
        """
        dir_path = os.path.join(output_dir, self.directory)
        os.makedirs(dir_path, exist_ok=True)
        if filename is None:
            filename = '_'.join([m.replace(' ', '_') for m in self.metrics]) + '.png'
        # remove any path separators from filename to prevent unintended directories
        filename = filename.replace('/', '_').replace('\\', '_')
        out_path = os.path.join(str(dir_path), filename)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved chart to {out_path}")
