class IPlotter:
    """Interface for plotters."""
    def plot(self, metrics, dfs, exp_names, output_dir):
        raise NotImplementedError("Plotter subclasses must implement the plot method.")
