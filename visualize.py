from visualization.bar_plotter import BarPlotter
from visualization.box_plotter import BoxPlotter
from visualization.experiment_lists import ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE_REFINER, \
    ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE, ML_EXPERIMENTS_ALL, SSL_BEST_REFINER, SSL_3_BEST
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.line_plotter import LinePlotter
from visualization.scatter_plotter import ScatterPlotter
from visualization.visualizer import Visualizer

# ----------------- Configuration Section -----------------
BASE_VALIDATION_PATH = 'data/Results/Validation'

OUTPUT_DIR = 'data/Results/Plots/New_Plots'
# Define plotter instances to use
# Mapping für die Labels:
experiment_labels = {
    exp: ('mit ColorPatchRefiner' if exp in ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE_REFINER else 'ohne ColorPatchRefiner')
    for exp in ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE
}
PLOTTERS = [
    BarPlotter(ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE, ['Dice']),
    LinePlotter(ML_EXPERIMENTS_ALL, ['Dice']),
    ScatterPlotter(SSL_BEST_REFINER, ['Precision', 'Recall'], show_legend=False),
    HeatmapPlotter(SSL_3_BEST, ['Dice']),
    BoxPlotter(SSL_3_BEST, ['Dice']),
]

# ---------------------------------------------------------------





def main():
    # Instantiate and run Visualizer with configuration above
    viz = Visualizer(BASE_VALIDATION_PATH, PLOTTERS, OUTPUT_DIR)
    viz.visualize()


if __name__ == '__main__':
    main()
