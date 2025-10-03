from visualization.atlas_experiment_lists import ATLAS_EXPERIMENTS_ATLAS_DATA, ATLAS_EXPERIMENTS_ATLAS_DATA_REFINER, \
    ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE, ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE_REFINER, \
    ATLAS_EXPERIMENTS_ALL, ATLAS_REFINER_ALL, ATLAS_NO_REFINER_ALL, ATLAS_REFINER_TH_3, ATLAS_REFINER_TH_5, ATLAS_BEST, \
    ATLAS_EXPERIMENTS_OVER_60_DICE, ML_BEST, ML_BEST_REFINE, ML_EXPERIMENTS_ALL, ML_WITH_REFINER, ML_WITHOUT_REFINER, \
    ML_3_BEST, OVERALL_3_BEST, ATLAS_3_BEST, SCHULTER, ATLAS_BEST_NO_REFINER, SSL_ALL_TORSO_NO_REFINER, \
    SSL_ALL_TORSO_REFINER, SSL_BEST, SSL_BEST_REFINER, SSL_3_BEST
from visualization.bar_plotter import BarPlotter
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.line_plotter import LinePlotter
from visualization.metric_count_bar_plotter import MetricCountBarPlotter
from visualization.scatter_plotter import ScatterPlotter
from visualization.visualizer import Visualizer

# ----------------- Configuration Section -----------------
BASE_VALIDATION_PATH = 'data/Results/Validation'

OUTPUT_DIR = 'data/Results/Plots/New'
# Define plotter instances to use
# Mapping für die Labels:
experiment_labels = {
    exp: ('mit ColorPatchRefiner' if exp in ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE_REFINER else 'ohne ColorPatchRefiner')
    for exp in ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE
}
PLOTTERS = [
    # absolute counts per metric
    # BarPlotter(
    #     experiments=ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE,
    #     metrics=['Dice'],
    #     highlighted_experiments=ATLAS_EXPERIMENTS_ATLAS_DATA_BMI_PERCENTILE_REFINER,
    #     experiment_labels=experiment_labels
    # )
    # BarPlotter(SCHULTER, ['Dice']),
    # BarPlotter(SSL_ALL_TORSO_NO_REFINER, ['Average duration per image']),
    # BarPlotter(SCHULTER, ['Average duration per image'], show_ms_in_duration=True)
    # LinePlotter(ML_EXPERIMENTS_ALL, ['Dice']),
    # ScatterPlotter(SSL_BEST_REFINER, ['Precision', 'Recall'], show_legend=False),
    # MetricCountBarPlotter(ATLAS_BEST, ['Segments Center Success (tol=3mm)', 'GT Center Angle DIERS Success (tol=4.2°)']),
    # BubblePlotter(ATLAS_EXPERIMENTS_ALL, ['N Segments GT', 'Dice']),
    # AvgDiceByNSegmentsPlotter(ATLAS_EXPERIMENTS_ALL, ['N Segments GT', 'Dice']),
    # BoxDiceByNSegmentsPlotter(ATLAS_EXPERIMENTS_ALL, ['N Segments GT', 'Dice']),
    # ScatterPlotter(ATLAS_BEST, ['Segments Center Angle Error [degrees]', 'Dice']),
    ScatterPlotter([SCHULTER[1]], ['Precision', 'Recall'], show_legend=False),
    # ScatterPlotter(ML_BEST, ['Segments Center Success (tol=3mm)', 'Dice']),
    # HeatmapPlotter(SSL_3_BEST, ['Dice'])
    # ScatterPlotter(SSL_BEST, ['N Segments GT', 'Dice'], show_legend=False)
]

# ---------------------------------------------------------------





def main():
    # Instantiate and run Visualizer with configuration above
    viz = Visualizer(BASE_VALIDATION_PATH, PLOTTERS, OUTPUT_DIR)
    viz.visualize()


if __name__ == '__main__':
    main()
