from skimage import io


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = io.imread(self.image_path)
        self.preprocessing_parameters = []

    def append_preprocessing_parameters(self, parameters):
        self.preprocessing_parameters.append(parameters)
