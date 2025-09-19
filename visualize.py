from visualization.bar_plotter import BarPlotter
from visualization.box_plotter import BoxPlotter
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.line_plotter import LinePlotter
from visualization.scatter_plotter import ScatterPlotter
from visualization.visualizer import Visualizer

# ----------------- Configuration Section -----------------
BASE_VALIDATION_PATH = 'data/Results/Validation_old'
EXPERIMENTS = [
    'Atlas_Experiment01',
    'Atlas_Experiment02',
    'Atlas_Experiment03',
    'Atlas_Experiment04',
    'Atlas_Experiment05',
    'Atlas_Experiment06',
    'Atlas_Experiment07',
    'Atlas_Experiment08',
    'Atlas_Experiment09',
    'Atlas_Experiment10',
    'Atlas_Experiment11',
    'Atlas_Experiment12',
    'Atlas_Experiment13',
    'Atlas_Experiment14',
    'Atlas_Experiment15',
    'Atlas_Experiment16',
    'Atlas_Experiment17',
    'Atlas_Experiment18',
    'Atlas_Experiment19',
    'Atlas_Experiment20',
    'Atlas_Experiment21',
    'Atlas_Experiment22',
    'Atlas_Experiment23',
    'Atlas_Experiment24',
    'Atlas_Experiment25',
    'Atlas_Experiment26',
    'Atlas_Experiment27',
    'Atlas_Experiment28',
    'Atlas_Experiment29',
    'Atlas_Experiment30',
    'Atlas_Experiment31',
    'Atlas_Experiment32',
    'Atlas_Experiment33',
    'Atlas_Experiment34',
    'Atlas_Experiment35',
    'Atlas_Experiment36',
    'Atlas_Experiment37',
    'Atlas_Experiment38',
    'Atlas_Experiment39',
    'Atlas_Experiment40',
    'Atlas_Experiment41',

    # 'Atlas_Experiment42',
    # 'Atlas_Experiment43',
    # 'Atlas_Experiment44',
    # 'Atlas_Experiment45',
    # 'Atlas_Experiment46',
    # 'Atlas_Experiment47',
    # 'Atlas_Experiment48',
    # 'Atlas_Experiment49',
    # 'Atlas_Experiment50',
    # 'Atlas_Experiment51',
    # 'Atlas_Experiment52',
    # 'Atlas_Experiment53',
    # 'Atlas_Experiment54',
    # 'Atlas_Experiment55',
    # 'Atlas_Experiment56',
    # 'Atlas_Experiment57',
    # 'Atlas_Experiment58',
    # 'Atlas_Experiment59',
    # 'Atlas_Experiment60',
    # 'Atlas_Experiment61',
    # 'Atlas_Experiment62',
    # 'Atlas_Experiment63',
    # 'Atlas_Experiment64',
    # 'Atlas_Experiment65',
    # 'Atlas_Experiment66',
    # 'Atlas_Experiment67',
    # 'Atlas_Experiment68',
    # 'Atlas_Experiment69',
    # 'Atlas_Experiment70',
    # 'Atlas_Experiment71',
    # 'Atlas_Experiment72',
    # 'Atlas_Experiment73',
    # 'Atlas_Experiment74',
    # 'Atlas_Experiment75',
    # 'Atlas_Experiment76',
    # 'Atlas_Experiment77',
    # 'Atlas_Experiment78',
    # 'Atlas_Experiment79',
    # 'Atlas_Experiment80',
    # 'Atlas_Experiment81',
    # 'Atlas_Experiment82',
]
METRICS = [
    # Column names to plot
    'Center Angle Success',
    # 'Center Pred Success',
    # 'Recall',
    # 'Center Pred Success'
]
OUTPUT_DIR = 'data/Results/Plots/Atlas_Experiments'
# Define plotter instances to use
PLOTTERS = [
    # BarPlotter(),
    # BoxPlotter(),
    # LinePlotter(),
    HeatmapPlotter(),
    # ScatterPlotter()
]
# TODO: Pro Bild was erstellt werden soll einen Plotter mit einzubeziehenden Metriken definieren (dann können auch mehrere Balken zusammen geplottet werden)
# ---------------------------------------------------------------





def main():
    # Instantiate and run Visualizer with configuration above
    viz = Visualizer(BASE_VALIDATION_PATH, EXPERIMENTS, PLOTTERS, METRICS, OUTPUT_DIR)
    viz.visualize()


if __name__ == '__main__':
    main()
