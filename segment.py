from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from atlas.selector.bmi_atlas_selector import BmiAtlasSelector
from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from atlas.voter.weighted_majority_voter import WeightedMajorityVoter
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from preprocessing.dimples_roi_preprocessor import DimplesRoiPreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor
from segmenter.atlas_segmenter import AtlasSegmenter
from segmenter.segmentation_runner import SegmentationRunner

ATLAS_DIR = "data/Atlas_Data"
ATLAS_DIR_BMI_PERCENTILE = "data/Atlas_Data_BMI_Percentile"
TARGET_IMAGES_DIR = "data/Validation_Data_Small"
IMAGE_INFO_PATH = "data/Info_Sheets/All_Data_Renamed_overview.csv"
BMI_TABLE_PATH = "data/Info_Sheets/bmi_table_who.csv"

image_segmenter = [
    # AtlasSegmenter(
    #     num_atlases_to_select=3,
    #     atlas_dir=ATLAS_DIR,
    #     preprocessing_steps=[],
    #     atlas_selector=SimilarityAtlasSelector(),
    #     segmentation_voter=MajorityVoter(),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment01"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR,
    #     preprocessing_steps=[],
    #     atlas_selector=SimilarityAtlasSelector(),
    #     segmentation_voter=MajorityVoter(),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment02"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR,
    #     preprocessing_steps=[],
    #     atlas_selector=SimilarityAtlasSelector(),
    #     segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment03"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR,
    #     preprocessing_steps=[],
    #     atlas_selector=SimilarityAtlasSelector(),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment04"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR,
    #     preprocessing_steps=[],
    #     atlas_selector=SimilarityAtlasSelector(),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment05"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=3,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=MajorityVoter(),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment06"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=MajorityVoter(),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment07"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment08"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment09"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment10"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment11"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment12"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment13"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment14"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment15"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment16"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment17"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment18"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment19"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment20"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment21"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment22"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment23"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment24"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment25"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment26"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment27"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment28"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment29"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=13,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment30"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment31"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment32"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment33"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment34"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=None,
    #     output_dir="data/Atlas_Experiment35"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment36"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7)],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment37"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment38"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment39"
    # ),
    # AtlasSegmenter(
    #     num_atlases_to_select=5,
    #     atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
    #     preprocessing_steps=[DimplesRoiPreprocessor(target_ratio=10/7), BlueColorPreprocessor()],
    #     atlas_selector=BmiAtlasSelector(image_info_path=IMAGE_INFO_PATH, bmi_table_path=BMI_TABLE_PATH),
    #     segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
    #     segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
    #     output_dir="data/Atlas_Experiment40"
    # )
    AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir="data/Atlas_Experiment41"
    )
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        SegmentationRunner(segmenter, TARGET_IMAGES_DIR).run()
