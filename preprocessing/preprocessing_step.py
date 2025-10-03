from abc import ABC, abstractmethod


class IPreprocessingStep(ABC):
    """
    Interface for preprocessing steps. All preprocessing step classes should inherit from this interface
    and implement the required methods for image and mask preprocessing.
    """

    @abstractmethod
    def preprocess_image(self, image):
        """
        Preprocess an image and return the processed image and parameters.
        Args:
            image (np.ndarray): Input image.
        Returns:
            tuple: (processed_image, parameters)
        """
        pass

    @abstractmethod
    def preprocess_mask(self, image, parameters):
        """
        Preprocess a mask using the same parameters as the corresponding image.
        Args:
            image (np.ndarray): Input mask.
            parameters (dict): Parameters from preprocess_image.
        Returns:
            np.ndarray: Preprocessed mask.
        """
        pass

    @abstractmethod
    def undo_preprocessing(self, preprocessed_image, parameters):
        """
        Reverse the preprocessing to restore the original image size and content.
        Args:
            preprocessed_image (np.ndarray): The preprocessed image.
            parameters (dict): Parameters from preprocess_image.
        Returns:
            np.ndarray: Restored image.
        """
        pass