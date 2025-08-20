from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from atlas.selector.bmi_atlas_selector import BmiAtlasSelector
from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from atlas.voter.weighted_majority_voter import WeightedMajorityVoter
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from preprocessing.color_preprocessor import ColorPreprocessor
from preprocessing.dimples_roi_preprocessor import DimplesRoiPreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor
from segmenter.atlas_segmenter import AtlasSegmenter


class AtlasSegmentationRunner:
    def __init__(self, num_atlases_to_select, atlas_dir, preprocessing_steps, atlas_selector, segmentation_voter, segmentation_refiner, output_dir, target_images_dir):
        self.segmenter = AtlasSegmenter(
            num_atlases_to_select,
            atlas_dir,
            preprocessing_steps,
            atlas_selector,
            segmentation_voter,
            segmentation_refiner,
            output_dir
        )
        self.target_images_dir = target_images_dir

    def run(self):
        target_images = self.segmenter.load_target_images(self.target_images_dir)
        segmented_images = self.segmenter.segment_images(target_images)
        self.segmenter.save_segmentation(segmented_images)

# Beispiel für die Ausführung:
if __name__ == "__main__":
    # Hier müssen die passenden Objekte und Parameter übergeben werden
    runner = AtlasSegmentationRunner(
        num_atlases_to_select=13,
        atlas_dir="data/Atlas_Data_BMI_Percentile",
        # preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7) ,BlueColorPreprocessor()],  # Liste mit Preprocessing-Objekten
        preprocessing_steps=[],  # Liste mit Preprocessing-Objekten
        atlas_selector=BmiAtlasSelector("data/Info_Sheets/All_Data_Renamed_overview.csv", "data/Info_Sheets/bmi_table_who.csv"),      # AtlasSelector-Objekt
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),  # SegmentationVoter-Objekt
        segmentation_refiner=ColorPatchRefiner(BlueColorPreprocessor()),
        output_dir="data/Atlas_Experiment100",
        target_images_dir="data/Validation_Data_Small"
    )
    # runner = AtlasSegmentationRunner(
    #     num_atlases_to_select=3,
    #     atlas_dir="data/Atlas_Data",
    #     preprocessing_steps=[],  # Liste mit Preprocessing-Objekten
    #     atlas_selector=SimilarityAtlasSelector(),      # AtlasSelector-Objekt
    #     segmentation_voter=MajorityVoter(),  # SegmentationVoter-Objekt
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment01",
    #     target_images_dir="data/Validation_Data_Small"
    # )
    runner.run()