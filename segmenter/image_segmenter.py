from abc import ABC, abstractmethod


class IImageSegmenter(ABC):
    @abstractmethod
    def load_target_images(self, directory_path: str) -> list[str]:
        pass

    @abstractmethod
    def segment_images(self, image_paths: list[str]):
        pass

    @abstractmethod
    def save_segmentation(self, segmented_images):
        pass
