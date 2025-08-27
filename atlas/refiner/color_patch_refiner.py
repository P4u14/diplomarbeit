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
        if target_segmentation.ndim == 2:
            target_segmentation_bool = target_segmentation > 0  # [H,W]
            was_2d = True
            C_in = 1
        elif target_segmentation.ndim == 3:
            was_2d = False
            C_in = target_segmentation.shape[2]
            if C_in == 1:
                target_segmentation_bool = target_segmentation[..., 0] > 0
            else:
                target_segmentation_bool = np.any(target_segmentation > 0, axis=2)  # [H,W]
        else:
            raise ValueError(f"Unexpected target_segmentation shape: {target_segmentation.shape}")

        target_color_bool = np.any(target_color_mask > 0, axis=2)  # [H,W]

        # Starting points: Intersection
        seed = target_segmentation_bool & target_color_bool

        # Region growing within target_color_bool (8-neighborhood)
        grown_region = binary_propagation(seed, mask=target_color_bool, structure=np.ones((3, 3), bool))

        # Result = original segmentation ∪ grown_region region
        refined_segmentation_bool = target_segmentation_bool | grown_region

        # Back in 3-channel form (1 white, 0 black)
        white_val = target_segmentation.max()
        if was_2d:
            refined = np.zeros_like(target_segmentation)
            refined[refined_segmentation_bool] = white_val
            return refined
        else:
            refined_segmentation = np.zeros_like(target_segmentation)
            if C_in == 1:
                refined_segmentation[..., 0][refined_segmentation_bool] = white_val
            else:
                for c in range(C_in):
                    refined_segmentation[..., c][refined_segmentation_bool] = white_val
            return refined_segmentation
