from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from segmenter.atlas_segmenter import AtlasSegmenter
from segmenter.segmentation_runner import SegmentationRunner

ATLAS_DIR = "data/Atlas_Data"
ATLAS_DIR_BMI_PERCENTILE = "data/Atlas_Data_BMI_Percentile"
TARGET_IMAGES_DIR = "data/Validation_Data_Small"

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
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        SegmentationRunner(segmenter, TARGET_IMAGES_DIR).run()
