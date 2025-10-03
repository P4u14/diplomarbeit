from abc import ABC, abstractmethod


class IAtlasSelector(ABC):
    @abstractmethod
    def select_atlases(self, atlases, target_image, n):
        pass


class AtlasSelector(IAtlasSelector):
    """
    Atlas selector that selects the first N atlases from the provided list.
    This is a simple baseline selector for segmentation workflows.
    """

    def select_atlases(self, atlases, target_image, num_atlases_to_select):
        """
        Select the first N atlases from the list.
        Args:
            atlases (list): List of available Atlas objects.
            target_image: The target image for which atlases are to be selected.
            num_atlases_to_select (int): Number of atlases to select.
        Returns:
            list: List of selected Atlas objects.
        """
        return atlases[:num_atlases_to_select]