from skimage import io

from segmenter.atlas.bmi_percentiles.bmi_percentile_calculator import BmiPercentileCalculator


class Atlas:
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image = io.imread(self.image_path)
        self.preprocessed_image = self.image.copy()
        self.preprocessing_parameters = []
        self.mask = io.imread(self.mask_path)
        self.preprocessed_mask = self.mask.copy()
        self.bmi_percentile_interval = None

    def append_preprocessing_parameters(self, parameters):
        self.preprocessing_parameters.append(parameters)

    def set_bmi_info(self, image_info_path, bmi_table_path):
        bmi_percentile_calculator = BmiPercentileCalculator(image_info_path, bmi_table_path)
        self.bmi_percentile_interval = bmi_percentile_calculator.calculate_bmi_percentile_interval(self.image_path)





