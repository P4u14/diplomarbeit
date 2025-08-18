import numpy as np

from preprocessing.color_preprocessor import ColorPreprocessor


class BlueColorPreprocessor(ColorPreprocessor):
    def __init__(self):
        lower_blue = np.array([100, 50, 30])
        upper_blue = np.array([130, 255, 255])
        super().__init__(lower_color=lower_blue, upper_color=upper_blue)
