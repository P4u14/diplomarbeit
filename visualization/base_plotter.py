import os

from matplotlib import pyplot as plt

from visualization.iplotter import IPlotter


class BasePlotter(IPlotter):
    def __init__(self, experiments, metrics, directory):
        self.experiments = experiments
        self.metrics = metrics
        self.directory = directory

    def plot(self, data_frames, output_dir):
        raise NotImplementedError("Subclasses must implement the plot method.")

    def save_plot(self, fig, output_dir, filename=None):
        """Save figure in the configured directory, using optional filename."""
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
