from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from atlas.voter.weighted_majority_voter import WeightedMajorityVoter
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from segmenter.atlas_segmenter import AtlasSegmenter

ATLAS_DIR = "data/Images/Atlas_Data"
ATLAS_BASE_OUTPUT_DIR = "data/Results/Segmentation_Results/Atlas/"

atlas_experiments = {
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
        output_dir=ATLAS_BASE_OUTPUT_DIR + "Atlas_Experiment12"
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
    )
}