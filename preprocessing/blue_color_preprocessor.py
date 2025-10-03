import numpy as np

from preprocessing.color_preprocessor import ColorPreprocessor


class BlueColorPreprocessor(ColorPreprocessor):
    """
    Preprocessor for extracting blue color regions from images using HSV color space thresholds.
    Inherits from ColorPreprocessor and sets the lower and upper bounds for blue color detection.
    """

    def __init__(self):
        """
        Initialize the BlueColorPreprocessor with predefined HSV bounds for blue color.
        """
        lower_blue = np.array([100, 50, 30])
        upper_blue = np.array([130, 255, 255])
        super().__init__(lower_color=lower_blue, upper_color=upper_blue)
