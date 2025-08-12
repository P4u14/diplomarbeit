from skimage import io


class Atlas:
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image = io.imread(self.image_path)
        self.preprocessed_image = self.image # TODO
        self.mask = io.imread(self.mask_path)
        self.preprocessed_mask = self.mask # TODO
        self.preprocessing_steps = []


