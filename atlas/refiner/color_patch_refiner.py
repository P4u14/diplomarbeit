import numpy as np
from scipy.ndimage import binary_propagation

from atlas.refiner.segmentation_refiner import ISegmentationRefiner


class ColorPatchRefiner(ISegmentationRefiner):

    def __init__(self, color_preprocessor):
        self.color_preprocessor = color_preprocessor

    def refine(self, target_segmentation, target_image):
        if target_segmentation.max() == 0:
            return target_segmentation

        # Get the color range mask and set it to 3 channels (0/1)
        target_color_mask, _ = self.color_preprocessor.preprocess_image(target_image.image)
        target_color_mask = (np.repeat(target_color_mask[:, :, np.newaxis], 3, axis=2) // 255).astype(np.uint8)

        # 2D boolean masks
        target_segmentation_bool = np.any(target_segmentation > 0, axis=2)  # [H,W]
        target_color_bool = np.any(target_color_mask > 0, axis=2)  # [H,W]

        # Starting points: Intersection
        seed = target_segmentation_bool & target_color_bool

        # Region growing within target_color_bool (8-neighborhood)
        grown_region = binary_propagation(seed, mask=target_color_bool, structure=np.ones((3, 3), bool))

        # Result = original segmentation ∪ grown_region region
        refined_segmentation_bool = target_segmentation_bool | grown_region

        # Back in 3-channel form (1 white, 0 black)
        refined_segmentation = np.zeros_like(target_segmentation)
        white_val = np.array(1, dtype=refined_segmentation.dtype)
        for c in range(3):
            refined_segmentation[..., c][refined_segmentation_bool] = white_val

        return refined_segmentation


