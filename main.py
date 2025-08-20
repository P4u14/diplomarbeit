from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from atlas.selector.bmi_atlas_selector import BmiAtlasSelector
from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from atlas.voter.weighted_majority_voter import WeightedMajorityVoter
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from preprocessing.color_preprocessor import ColorPreprocessor
from preprocessing.dimples_roi_preprocessor import DimplesRoiPreprocessor
from segmenter.atlas_segmenter import AtlasSegmenter
from segmenter.segmentation_runner import SegmentationRunner

ATLAS_DIR = "data/Atlas_Data"
ATLAS_DIR_BMI_PERCENTILE = "data/Atlas_Data_BMI_Percentile"
TARGET_IMAGES_DIR = "data/Validation_Data_Small"
IMAGE_INFO_PATH = "data/Info_Sheets/All_Data_Renamed_overview.csv"
BMI_TABLE_PATH = "data/Info_Sheets/bmi_table_who.csv"

image_segmenter = [
    AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir="data/Atlas_Experiment01"
    ),
    AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir="data/Atlas_Experiment02"
    ),
    AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
        atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir="data/Atlas_Experiment23"
    ),
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        SegmentationRunner(segmenter, TARGET_IMAGES_DIR).run()
