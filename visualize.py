from visualization.bar_plotter import BarPlotter
from visualization.box_plotter import BoxPlotter
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.line_plotter import LinePlotter
from visualization.scatter_plotter import ScatterPlotter
from visualization.visualizer import Visualizer

# --- Configuration: define base path, experiment names and output dir here ---
BASE_VALIDATION_PATH = 'data/Results/Validation'
EXPERIMENTS = [
    # Experiment identifiers (without suffix)
    'Atlas_Experiment75',
    'Atlas_Experiment78',
    'Atlas_Experiment79',
    'Atlas_Experiment80',
    'Atlas_Experiment81',
    'Atlas_Experiment82',
]
METRICS = [
    # Column names to plot
    'Dice',
    'Center Pred Success'
]
OUTPUT_DIR = 'data/Results/Plots/Test'
# Define plotter instances to use
PLOTTERS = [
    BarPlotter(),
    BoxPlotter(),
    LinePlotter(),
    HeatmapPlotter(),
    ScatterPlotter()
]
# ---------------------------------------------------------------





def main():
    # Instantiate and run Visualizer with configuration above
    viz = Visualizer(BASE_VALIDATION_PATH, EXPERIMENTS, PLOTTERS, METRICS, OUTPUT_DIR)
    viz.visualize()


if __name__ == '__main__':
    main()
