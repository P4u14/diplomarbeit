from atlas.bmi_percentiles.bmi_percentile_calculator import BmiPercentileCalculator
from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector


class BmiAtlasSelector(SimilarityAtlasSelector):

    def __init__(self, image_info_path, bmi_table_path):
        super().__init__()
        self.image_info_path = image_info_path
        self.bmi_table_path = bmi_table_path

    def select_atlases(self, atlases, target_image, n):
        preselected_atlases = self.preselect_atlases_on_bmi(atlases, target_image)
        scored_atlases = self.score_atlases(preselected_atlases, target_image)
        return scored_atlases[:n]

    def preselect_atlases_on_bmi(self, atlases, target_image):
        preselected_atlases = []
        bmi_percentile_calculator = BmiPercentileCalculator(self.image_info_path, self.bmi_table_path)
        target_bmi_percentile_intervall = bmi_percentile_calculator.calculate_bmi_percentile_interval(
            target_image.image_path)

        for atlas in atlases:
            if atlas.bmi_percentile_interval is None:
                atlas.set_bmi_info(self.image_info_path, self.bmi_table_path)
            if atlas.bmi_percentile_interval == target_bmi_percentile_intervall:
                preselected_atlases.append(atlas)

        if not preselected_atlases:
            print("No atlases selected for BMI percentile interval:", target_bmi_percentile_intervall)
            return atlases
        return preselected_atlases