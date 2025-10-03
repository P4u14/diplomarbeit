from abc import ABC, abstractmethod
from skimage import color
from skimage.util import img_as_float


class BaseAtlasSelector(ABC):
    """
    Abstract base class for atlas selection strategies in segmentation workflows.
    Subclasses must implement the select_atlases method to define how atlases are chosen for a given target image.
    """

    @abstractmethod
    def select_atlases(self, atlases, target_image, num_atlases_to_select):
        """
        Select a subset of atlases for a given target image.
        Args:
            atlases (list): List of available Atlas objects.
            target_image: The target image for which atlases are to be selected.
            num_atlases_to_select (int): Number of atlases to select.
        Returns:
            list: List of selected Atlas objects.
        """
        pass

    @staticmethod
    def to_gray(img):
        # RGBA -> RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = color.rgba2rgb(img)
        # RGB -> Gray
        if img.ndim == 3 and img.shape[2] == 3:
            img = color.rgb2gray(img)
        # To float [0, 1]
        img = img_as_float(img)
        return img