from abc import ABC, abstractmethod


class IImageSegmenter(ABC):
    @abstractmethod
    def load_target_images(self, directory_path):
        pass

    @abstractmethod
    def segment_images(self, image_paths):
        pass

    @abstractmethod
    def save_segmentation(self, segmented_image):
        pass
