from abc import ABC, abstractmethod


class ISegmentationRefiner(ABC):
    @abstractmethod
    def refine(self, target_segmentation, target_image):
        pass