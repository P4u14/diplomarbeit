import os
import matplotlib.pyplot as plt
from pandas import DataFrame


class Plotter:
    """Base class for plotters."""
    def plot(self, dfs: list[DataFrame], exp_names: list[str], output_dir: str):
        raise NotImplementedError("Plotter subclasses must implement the plot method.")
