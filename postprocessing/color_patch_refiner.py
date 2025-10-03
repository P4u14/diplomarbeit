import numpy as np
from scipy.ndimage import binary_propagation

from postprocessing.segmentation_refiner import ISegmentationRefiner


class ColorPatchRefiner(ISegmentationRefiner):
    """
    Refines segmentation masks by growing regions within a specified color range using region growing (binary propagation).
    The color range is defined by a color preprocessor. The refinement ensures that the segmentation includes contiguous
    regions matching the color mask, starting from the intersection of the original segmentation and the color mask.

    Args:
        color_preprocessor: An instance of a color preprocessor providing a preprocess_image method for color masking.
    """

    def __init__(self, color_preprocessor):
        """
        Initialize the ColorPatchRefiner.
        Args:
            color_preprocessor: Preprocessor instance for extracting the relevant color mask from images.
        """
        self.color_preprocessor = color_preprocessor

    def refine(self, target_segmentation, target_image):
        """
        Refine a segmentation mask by growing regions within the color mask using region growing.
        Args:
            target_segmentation (np.ndarray): The initial segmentation mask (2D or 3D array).
            target_image: An object with an 'image' attribute (the original image as np.ndarray).
        Returns:
            np.ndarray: The refined segmentation mask, same shape as input.
        """
        if target_segmentation.max() == 0:
            return target_segmentation

        # Get the color range mask and set it to 3 channels (0/1)
        target_color_mask, _ = self.color_preprocessor.preprocess_image(target_image.image)
        orig_colors = target_image.image[..., :3]
        colored_mask = np.zeros_like(orig_colors)
        mask_bool = target_color_mask > 0
        colored_mask[mask_bool] = orig_colors[mask_bool]

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
        # refined_segmentation_bool = target_segmentation_bool | grown_region
        refined_segmentation_bool = grown_region

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
