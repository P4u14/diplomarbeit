class IPlotter:
    """Interface for plotters."""
    def plot(self, dfs, output_dir):
        raise NotImplementedError("Plotter subclasses must implement the plot method.")
