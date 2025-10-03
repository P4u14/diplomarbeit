import os.path

from tqdm import tqdm

from segmenter.atlas.atlas import Atlas
from segmenter.base_segmenter import BaseSegmenter
from target_image.target_image import TargetImage
from target_image.target_segmentation import TargetSegmentation


class AtlasSegmenter(BaseSegmenter):

    def __init__(self, num_atlases_to_select, atlas_dir, preprocessing_steps, atlas_selector, segmentation_voter, segmentation_refiner, output_dir):
        super().__init__(output_dir, preprocessing_steps, segmentation_refiner)
        self.num_atlases_to_select = num_atlases_to_select
        self.atlas_dir = atlas_dir
        self.atlas_selector = atlas_selector
        self.segmentation_voter = segmentation_voter

    def load_atlases(self):
        atlases = []
        for file in os.listdir(self.atlas_dir):
            if file.endswith("-mask.Gauss" + self.img_extension):
                mask_path = os.path.join(self.atlas_dir, file)
                prefix = file[:-15]
                image_path = os.path.join(self.atlas_dir, prefix + ".Gauss.png")
                atlases.append(Atlas(image_path, mask_path))
        return atlases

    def segment_images(self, target_images: list[TargetImage]):
        atlases = self.load_atlases()

        # Preprocess atlases
        for atlas in tqdm(atlases, desc='Preprocessing atlases'):
            for pp_step in self.preprocessing_steps:
                atlas.preprocessed_image, parameters = pp_step.preprocess_image(atlas.preprocessed_image)
                atlas.append_preprocessing_parameters(parameters)
                atlas.preprocessed_mask = pp_step.preprocess_mask(atlas.preprocessed_mask, parameters)

        for target_image in tqdm(target_images, desc='Processed validation images'):
            # Preprocessing
            for pp_step in self.preprocessing_steps:
                target_image.preprocessed_image, parameters = pp_step.preprocess_image(target_image.preprocessed_image)
                target_image.append_preprocessing_parameters(parameters)

            # Atlas selection
            selected_atlases = self.atlas_selector.select_atlases(atlases, target_image, self.num_atlases_to_select)

            # Segmentation voting
            target_mask = self.segmentation_voter.vote(selected_atlases)

            # Undo-Preprocessing (reversed)
            for pp_step, parameters in reversed(list(zip(self.preprocessing_steps, target_image.preprocessing_parameters))):
                # pp_step.undo_preprocessing(target_image.preprocessed_image, parameters, True)
                target_mask = pp_step.undo_preprocessing(target_mask, parameters)

            # Segmentation refinement (optional)
            if self.segmentation_refiner is not None:
                target_mask = self.segmentation_refiner.refine(target_mask, target_image)

            # Save segmentation
            target_segmentation_path = os.path.basename(target_image.image_path)[:-10] + "-mask.Gauss.png"
            self.save_segmentation(TargetSegmentation(target_segmentation_path, target_mask))
