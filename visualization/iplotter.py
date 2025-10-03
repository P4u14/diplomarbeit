"""
iplotter.py - Interface definition for plotter classes.

This module defines the IPlotter interface, which specifies the required method for all plotter classes in the visualization package.

Classes:
    IPlotter: Abstract base class for plotters, requiring implementation of the plot method.
"""

class IPlotter:
    """
    Interface for plotter classes.

    All plotter classes should inherit from IPlotter and implement the plot method.

    Methods:
        plot(dfs, output_dir): Abstract method to generate plots from dataframes and save them to the output directory.
    """
    def plot(self, dfs, output_dir):
        """
        Abstract method to generate plots from the provided dataframes and save them to the specified output directory.

        Args:
            dfs (list): List of pandas DataFrames containing the data to plot.
            output_dir (str): Directory where the generated plots should be saved.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Plotter subclasses must implement the plot method.")
