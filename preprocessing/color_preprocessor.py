import cv2

from preprocessing.preprocessing_step import IPreprocessingStep


class ColorPreprocessor(IPreprocessingStep):
    def __init__(self, lower_color, upper_color):
        self.lower_color = lower_color
        self.upper_color = upper_color

    def preprocess_image(self, image):
        rgb_image = image[..., :3]
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        result_mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        return result_mask, None

    def preprocess_mask(self, image, parameters):
        return image

    def undo_preprocessing(self, preprocessed_image, parameters, is_already_color=False):
        return preprocessed_image