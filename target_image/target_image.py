from skimage import io


class TargetImage:
    """
    Data structure representing a target image for segmentation tasks.
    Stores the image, its path, a preprocessed version, and a list of preprocessing parameters.

    Args:
        image_path (str): Path to the image file.
    """

    def __init__(self, image_path):
        """
        Initialize the TargetImage by loading the image and preparing storage for preprocessing info.
        Args:
            image_path (str): Path to the image file.
        """
        self.image_path = image_path
        self.image = io.imread(self.image_path)
        self.preprocessed_image = self.image.copy()
        self.preprocessing_parameters = []

    def append_preprocessing_parameters(self, parameters):
        """
        Append preprocessing parameters to the list for later undoing or analysis.
        Args:
            parameters (dict): Parameters used in a preprocessing step.
        """
        self.preprocessing_parameters.append(parameters)
