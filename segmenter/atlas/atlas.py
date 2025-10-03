from skimage import io

from segmenter.atlas.bmi_percentiles.bmi_percentile_calculator import BmiPercentileCalculator


class Atlas:
    """
    Data structure representing an atlas image and its corresponding mask for segmentation tasks.
    Stores the image, mask, and their preprocessed versions, as well as preprocessing parameters and BMI percentile information.

    Args:
        image_path (str): Path to the atlas image file.
        mask_path (str): Path to the corresponding mask file.
    """

    def __init__(self, image_path, mask_path):
        """
        Initialize the Atlas object by loading the image and mask, and preparing storage for preprocessing and BMI info.
        Args:
            image_path (str): Path to the atlas image file.
            mask_path (str): Path to the corresponding mask file.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.image = io.imread(self.image_path)
        self.preprocessed_image = self.image.copy()
        self.preprocessing_parameters = []
        self.mask = io.imread(self.mask_path)
        self.preprocessed_mask = self.mask.copy()
        self.bmi_percentile_interval = None

    def append_preprocessing_parameters(self, parameters):
        """
        Append preprocessing parameters to the list for later undoing or analysis.
        Args:
            parameters (dict): Parameters used in a preprocessing step.
        """
        self.preprocessing_parameters.append(parameters)

    def set_bmi_info(self, image_info_path, bmi_table_path):
        """
        Calculate and set the BMI percentile interval for the atlas image using external info and a BMI table.
        Args:
            image_info_path (str): Path to the image info CSV or data file.
            bmi_table_path (str): Path to the BMI percentile table file.
        """
        bmi_percentile_calculator = BmiPercentileCalculator(image_info_path, bmi_table_path)
        self.bmi_percentile_interval = bmi_percentile_calculator.calculate_bmi_percentile_interval(self.image_path)
