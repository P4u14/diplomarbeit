from skimage import io


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = io.imread(self.image_path)
