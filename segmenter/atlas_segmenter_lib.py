from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from atlas.selector.bmi_atlas_selector import BmiAtlasSelector
from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from atlas.voter.weighted_majority_voter import WeightedMajorityVoter
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from preprocessing.dimples_roi_preprocessor import DimplesRoiPreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor
from segmenter.atlas_segmenter import AtlasSegmenter

ATLAS_DIR = "data/Images/Atlas_Data"
ATLAS_DIR_BMI_PERCENTILE = "data/Images/Atlas_Data_BMI_Percentile"
IMAGE_INFO_PATH = "data/Info_Sheets/All_Data_Renamed_overview.csv"
BMI_TABLE_PATH = "data/Info_Sheets/bmi_table_who.csv"
ATLAS_BASE_OUTPUT_DIR = "data/Results/Segmentation_Results/Atlas/"

atlas_segmenter = {
    1: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment01"
    ),
    2: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment02"
    ),
    3: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment03"
    ),
    4: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment04"
    ),
    5: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment05"
    ),
    6: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment06"
    ),
    7: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment07"
    ),
    8: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment08"
    ),
    9: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment09"
    ),
    10: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment10"
    ),
    11: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment11"
    ),
    12: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment12"
    ),
    13: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment13"
    ),
    14: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment14"
    ),
    15: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment15"
    ),
    16: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment16"
    ),
    17: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment17"
    ),
    18: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment18"
    ),
    19: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment19"
    ),
    20: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment20"
    ),
    21: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment21"
    ),
    22: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment22"
    ),
    23: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment23"
    ),
    24: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment24"
    ),
    25: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment25"
    ),
    26: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment26"
    ),
    27: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment27"
    ),
    28: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment28"
    ),
    29: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment29"
    ),
    30: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment30"
    ),
    31: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment31"
    ),
    32: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment32"
    ),
    33: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment33"
    ),
    34: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment34"
    ),
    35: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment35"
    ),
    36: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment36"
    ),
    37:  AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio= 5 / 7)],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment37"
    ),
    38:  AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[DimplesRoiPreprocessor(target_ratio= 10 / 7)],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment38"
    ),
    39:  AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[BlueColorPreprocessor()],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment39"
    ),
    40:  AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio= 5 / 7), BlueColorPreprocessor()],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment40"
    ),
    41:  AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR,
        preprocessing_steps=[DimplesRoiPreprocessor(target_ratio= 10 / 7), BlueColorPreprocessor()],
        atlas_selector=SimilarityAtlasSelector(),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment41"
    ),
    42: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment42"
    ),
    43: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment43"
    ),
    44: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment44"
    ),
    45: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment45"
    ),
    46: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment46"
    ),
    47: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=MajorityVoter(),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment47"
    ),
    48: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment48"
    ),
    49: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment49"
    ),
    50: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment50"
    ),
    51: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment51"
    ),
    52: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment52"
    ),
    53: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="normalize"),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment53"
    ),
    54: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment54"
    ),
    55: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment55"
    ),
    56: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment56"
    ),
    57: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment57"
    ),
    58: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment58"
    ),
    59: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment59"
    ),
    60: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment60"
    ),
    61: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment61"
    ),
    62: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment62"
    ),
    63: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment63"
    ),
    64: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment64"
    ),
    65: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=None,
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment65"
    ),
    66: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment66"
    ),
    67: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment67"
    ),
    68: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment68"
    ),
    69: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment69"
    ),
    70: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment70"
    ),
    71: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.02, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment71"
    ),
    72: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment72"
    ),
    73: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment73"
    ),
    74: AtlasSegmenter(
        num_atlases_to_select=74,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.5),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment74"
    ),
    75: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment75"
    ),
    76: AtlasSegmenter(
        num_atlases_to_select=5,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment76"
    ),
    77: AtlasSegmenter(
        num_atlases_to_select=13,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment77"
    ),
    78: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio= 5 / 7)],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment78"
    ),
    79: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[DimplesRoiPreprocessor(target_ratio= 10 / 7)],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment79"
    ),
    80: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[BlueColorPreprocessor()],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment80"
    ),
    81: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio= 5 / 7), BlueColorPreprocessor()],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment81"
    ),
    82: AtlasSegmenter(
        num_atlases_to_select=3,
        atlas_dir=ATLAS_DIR_BMI_PERCENTILE,
        preprocessing_steps=[DimplesRoiPreprocessor(target_ratio= 10 / 7), BlueColorPreprocessor()],
        atlas_selector=BmiAtlasSelector(bmi_table_path=BMI_TABLE_PATH, image_info_path=IMAGE_INFO_PATH),
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=0.2, threshold=0.3),
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment82"
    ),
}