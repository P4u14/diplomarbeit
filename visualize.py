from visualization.atlas_experiment_lists import ATLAS_EXPERIMENTS_ATLAS_DATA
from visualization.bar_plotter import BarPlotter
from visualization.box_plotter import BoxPlotter
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.line_plotter import LinePlotter
from visualization.scatter_plotter import ScatterPlotter
from visualization.visualizer import Visualizer

# ----------------- Configuration Section -----------------
BASE_VALIDATION_PATH = 'data/Results/Validation_old'

OUTPUT_DIR = 'data/Results/Plots/Atlas_Experiments_new'
# Define plotter instances to use
PLOTTERS = [
    HeatmapPlotter(ATLAS_EXPERIMENTS_ATLAS_DATA, ['Dice', 'N Segments Success']),
    HeatmapPlotter(ATLAS_EXPERIMENTS_ATLAS_DATA, ['Recall', 'Precision']),
    # BoxPlotter(),
    # LinePlotter(),
    # HeatmapPlotter(),
    # ScatterPlotter()
]
# TODO: Pro Bild was erstellt werden soll einen Plotter mit einzubeziehenden Metriken definieren (dann können auch mehrere Balken zusammen geplottet werden)
# ---------------------------------------------------------------





def main():
    # Instantiate and run Visualizer with configuration above
    viz = Visualizer(BASE_VALIDATION_PATH, PLOTTERS, OUTPUT_DIR)
    viz.visualize()


if __name__ == '__main__':
    main()
