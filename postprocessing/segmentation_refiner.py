from abc import ABC, abstractmethod


class ISegmentationRefiner(ABC):
    """
    Interface for segmentation refinement steps.
    All segmentation refiner classes should inherit from this interface and implement the refine method.
    """

    @abstractmethod
    def refine(self, target_segmentation, target_image):
        """
        Refine a segmentation mask based on the original image or additional information.
        Args:
            target_segmentation (np.ndarray): The initial segmentation mask to be refined.
            target_image: The original image or an object containing the image and metadata.
        Returns:
            np.ndarray: The refined segmentation mask.
        """
        pass