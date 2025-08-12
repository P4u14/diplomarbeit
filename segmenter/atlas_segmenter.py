import os.path

import numpy as np
from skimage import io
from tqdm import tqdm

from atlas.atlas import Atlas
from segmenter.image_segmenter import IImageSegmenter
from target_image.target_image import TargetImage, TargetSegmentation


class AtlasSegmenter(IImageSegmenter):
    IMG_EXTENSION = ".png"

    def __init__(self, num_atlases_to_select, atlas_dir, preprocessing_steps, atlas_selector, segmentation_voter, output_dir):
        self.num_atlases_to_select = num_atlases_to_select
        self.atlas_dir = atlas_dir
        self.preprocessing_steps = preprocessing_steps
        self.atlas_selector = atlas_selector
        self.segmentation_voter = segmentation_voter
        self.output_dir = output_dir

    def load_target_images(self, directory_path) -> list[str]:
        target_images = []
        for file in os.listdir(directory_path):
            if file.endswith(self.IMG_EXTENSION) and "-mask" not in file:
                target_images.append(TargetImage(os.path.join(directory_path, file)))
        return target_images

    def load_atlases(self) -> list[Atlas]:
        atlases = []
        for file in os.listdir(self.atlas_dir):
            if file.endswith("-mask.Gauss" + self.IMG_EXTENSION):
                mask_path = os.path.join(self.atlas_dir, file)
                prefix = file[:-15]
                image_path = os.path.join(self.atlas_dir, prefix + ".Gauss.png")
                atlases.append(Atlas(image_path, mask_path))
        return atlases

    def segment_images(self, target_images):
        atlases = self.load_atlases()

        for pp_step in self.preprocessing_steps:
            atlases = [pp_step.preprocess(atlas.preprocessed_image) for atlas in atlases]
            atlases = [pp_step.preprocess(atlas.preprocessed_mask) for atlas in atlases]

        for target_img in tqdm(target_images, desc='Processed validation images'):
            img = target_img.image

            for pp_step in self.preprocessing_steps:
                img = pp_step.preprocess(img)

            selected_atlases = self.atlas_selector.select_atlases(atlases, img, self.num_atlases_to_select)

            target_segmentation = self.segmentation_voter.vote(selected_atlases)

            for pp_step in reversed(self.preprocessing_steps):
                target_segmentation = pp_step.undo_preprocessing(target_segmentation)

            target_segmentation_path = os.path.basename(target_img.image_path)[:-10] + "-mask.Gauss.png"
            self.save_segmentation(TargetSegmentation(target_segmentation_path, target_segmentation))

    def save_segmentation(self, target_segmentation):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filepath = os.path.join(self.output_dir, target_segmentation.output_path)
        segmentation = target_segmentation.result_mask
        io.imsave(str(filepath), (segmentation * 255).astype(np.uint8))
        print("Saved segmentation mask to {}".format(filepath))




