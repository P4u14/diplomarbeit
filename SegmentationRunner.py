from atlas.selector.similarity_atlas_selector import SimilarityAtlasSelector
from atlas.voter.majority_voter import MajorityVoter
from atlas.voter.weighted_majority_voter import WeightedMajorityVoter
from segmenter.atlas_segmenter import AtlasSegmenter


class AtlasSegmentationRunner:
    def __init__(self, num_atlases_to_select, atlas_dir, preprocessing_steps, atlas_selector, segmentation_voter, output_dir, target_images_dir):
        self.segmenter = AtlasSegmenter(
            num_atlases_to_select,
            atlas_dir,
            preprocessing_steps,
            atlas_selector,
            segmentation_voter,
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
        num_atlases_to_select=3,
        atlas_dir="data/Atlas_Data",
        preprocessing_steps=[],  # Liste mit Preprocessing-Objekten
        atlas_selector=SimilarityAtlasSelector(),      # AtlasSelector-Objekt
        segmentation_voter=WeightedMajorityVoter(scheme="softmax", temperature=1),  # SegmentationVoter-Objekt
        output_dir="data/Atlas_Experiment04",
        target_images_dir="data/Validation_Data_Small"
    )
    runner.run()