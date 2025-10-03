import numpy as np
from skimage.transform import resize

from preprocessing.preprocessing_step import IPreprocessingStep


class TorsoRoiPreprocessor(IPreprocessingStep):
    """
    Preprocessing step for extracting, padding, and resizing the torso region of interest (ROI) from an image.
    This class provides methods to crop the torso, pad to a target aspect ratio, rescale, and reverse these operations.
    """

    def __init__(self, target_ratio):
        """
        Initialize the TorsoRoiPreprocessor.
        Args:
            target_ratio (float): Desired width-to-height ratio for the output image.
        """
        self.target_ratio = target_ratio  # width : height

    def preprocess_image(self, image):
        """
        Preprocess an image by cropping the torso ROI, padding to the target ratio, and resizing to the original size.
        Args:
            image (np.ndarray): Input image.
        Returns:
            tuple: (resized_image, parameters) where parameters is a dict with crop, pad, and size info.
        """
        original_size = {
            'height': image.shape[0],
            'width': image.shape[1]
        }
        cropped_image, bbox = self.crop_torso_roi(image)
        cropped_and_padded_image, padding, padded_size = self.pad_image_to_correct_ratio(cropped_image, bbox)
        resized_image = self.rescale_image(cropped_and_padded_image, original_size)
        parameters = {
            'original_size': original_size,
            'bbox': bbox,
            'padding': padding,
            'padded_size': padded_size,
        }
        return resized_image, parameters

    def preprocess_mask(self, image, parameters):
        """
        Preprocess a mask using the same parameters as the corresponding image.
        Args:
            image (np.ndarray): Input mask.
            parameters (dict): Parameters from preprocess_image.
        Returns:
            np.ndarray: Preprocessed mask.
        """
        cropped_image = self.crop_with_parameters(image, parameters['bbox'])
        cropped_and_padded_image = self.pad_image_with_parameters(cropped_image, parameters['padding'])
        resized_image = self.rescale_image(cropped_and_padded_image, parameters['original_size'])
        return resized_image

    def undo_preprocessing(self, preprocessed_image, parameters, is_already_color=False):
        """
        Reverse the preprocessing to restore the original image size and content.
        Args:
            preprocessed_image (np.ndarray): The preprocessed image.
            parameters (dict): Parameters from preprocess_image.
            is_already_color (bool): If True, treat as color image.
        Returns:
            np.ndarray: Restored image.
        """
        cropped_and_padded_image = self.undo_rescale_image(preprocessed_image, parameters['padded_size'])
        cropped_image = self.undo_pad_image_to_correct_ratio(cropped_and_padded_image, parameters['padding'])
        image = self.undo_crop_torso_roi(cropped_image, parameters['original_size'], parameters['bbox'])
        return image

    @staticmethod
    def crop_torso_roi(image):
        """
        Crop the torso region of interest (ROI) from the image based on non-black pixels.
        Args:
            image (np.ndarray): Input image.
        Returns:
            tuple: (cropped_image, bbox) where bbox is a dict with crop coordinates.
        """
        image_rgb = image[..., :3]
        torso_roi = np.any(image_rgb != [0, 0, 0], axis=-1)
        coords = np.argwhere(torso_roi)
        if coords.size == 0:
            raise ValueError("No torso region found in the image.")
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        bbox = {
            'min_x': min_x,
            'max_x': max_x + 1,  # use exclusive values
            'min_y': min_y,
            'max_y': max_y + 1  # use exclusive values
        }
        cropped_image = image[min_y:max_y, min_x:max_x, ...]
        return cropped_image, bbox

    @staticmethod
    def crop_with_parameters(image, bbox):
        """
        Crop an image using a bounding box.
        Args:
            image (np.ndarray): Input image.
            bbox (dict): Bounding box with min/max x/y.
        Returns:
            np.ndarray: Cropped image.
        """
        cropped_image = image[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], ...]
        return cropped_image

    @staticmethod
    def undo_crop_torso_roi(cropped_image, original_size, bbox):
        """
        Restore a cropped image to its original size using the bounding box.
        Args:
            cropped_image (np.ndarray): Cropped image.
            original_size (dict): Original image size.
            bbox (dict): Bounding box used for cropping.
        Returns:
            np.ndarray: Restored image.
        """
        H, W = original_size['height'], original_size['width']

        if cropped_image.ndim == 3:
            C = cropped_image.shape[2]
            restored = np.zeros((H, W, C), dtype=cropped_image.dtype)
            restored[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], :] = cropped_image
        elif cropped_image.ndim == 2:
            restored = np.zeros((H, W), dtype=cropped_image.dtype)
            restored[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x']] = cropped_image
        else:
            raise ValueError(f"Unsupported ndim {cropped_image.ndim} in undo_crop_torso_roi")
        return restored

    def pad_image_to_correct_ratio(self, cropped_image, bbox):
        """
        Pad the cropped image to achieve the target aspect ratio.
        Args:
            cropped_image (np.ndarray): Cropped image.
            bbox (dict): Bounding box used for cropping.
        Returns:
            tuple: (padded_image, padding, padded_size)
        """
        width = bbox['max_x'] - bbox['min_x']
        height = bbox['max_y'] - bbox['min_y']
        current_ratio = width / height

        if current_ratio < self.target_ratio:
            padded_width = int(round(height * self.target_ratio))
            padded_height = height
            pad_total = padded_width - width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_top = 0
            pad_bottom = 0
        elif current_ratio > self.target_ratio:
            padded_width = width
            padded_height = int(round(width / self.target_ratio))
            pad_total = padded_height - height
            pad_left = 0
            pad_right = 0
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        else:
            padded_width = width
            padded_height = height
            pad_left = pad_right = pad_top = pad_bottom = 0

        padding = {
            'left': pad_left,
            'right': pad_right,
            'top': pad_top,
            'bottom': pad_bottom
        }

        padded_size = {
            'width': padded_width,
            'height': padded_height
        }

        padded_image = np.pad(
            cropped_image,
            ((padding['top'], padding['bottom']), (padding['left'], padding['right']), (0, 0)),
            mode='constant',
            constant_values=0
        )
        return padded_image, padding, padded_size

    @staticmethod
    def pad_image_with_parameters(cropped_image, padding):
        """
        Pad an image using specified padding values.
        Args:
            cropped_image (np.ndarray): Cropped image.
            padding (dict): Padding values for each side.
        Returns:
            np.ndarray: Padded image.
        """
        padded_image = np.pad(
            cropped_image,
            ((padding['top'], padding['bottom']), (padding['left'], padding['right']), (0, 0)),
            mode='constant',
            constant_values=0
        )
        return padded_image

    @staticmethod
    def undo_pad_image_to_correct_ratio(padded_image, padding):
        """
        Remove padding from an image using specified padding values.
        Args:
            padded_image (np.ndarray): Padded image.
            padding (dict): Padding values for each side.
        Returns:
            np.ndarray: Unpadded image.
        """
        top, bottom = padding['top'], padding['bottom']
        left, right = padding['left'], padding['right']
        height, width = padded_image.shape[:2]
        unpadded_image = padded_image[
            top:height - bottom if bottom > 0 else height,
            left:width - right if right > 0 else width,
            ...
        ]
        return unpadded_image

    def rescale_image(self, cropped_and_padded_image, original_size):
        """
        Rescale the image to the original size (width, height) after padding.
        Args:
            cropped_and_padded_image (np.ndarray): Image after cropping and padding.
            original_size (dict): Original image size.
        Returns:
            np.ndarray: Rescaled image.
        """
        target_height = original_size['width'] / self.target_ratio
        output_shape = (target_height, original_size['width'], cropped_and_padded_image.shape[2])
        rescaled_image = resize(
            cropped_and_padded_image,
            output_shape,
            order=1,  # bilinear
            mode='constant',
            cval=0,
            anti_aliasing=True,
            preserve_range=True
        ).astype(cropped_and_padded_image.dtype)
        return rescaled_image

    @staticmethod
    def undo_rescale_image(rescaled_image, padded_size):
        """
        Reverse the rescaling to restore the padded image to its original padded size.
        Args:
            rescaled_image (np.ndarray): Rescaled image.
            padded_size (dict): Size after padding.
        Returns:
            np.ndarray: Unscaled image.
        """
        # Automatically detect image type (color, grayscale, or binary)
        is_binary = len(np.unique(rescaled_image)) <= 2

        if rescaled_image.ndim == 3:
            # Color image
            output_shape = (padded_size['height'], padded_size['width'], rescaled_image.shape[2])
            if is_binary:
                order = 0  # Nearest-neighbor for binary
                anti_aliasing = False
            else:
                order = 1  # Bilinear for grayscale
                anti_aliasing = True
        elif rescaled_image.ndim == 2:
            # Grayscale or binary image
            output_shape = (padded_size['height'], padded_size['width'])
            if is_binary:
                order = 0  # Nearest-neighbor for binary
                anti_aliasing = False
            else:
                order = 1  # Bilinear for grayscale
                anti_aliasing = True
        else:
            raise ValueError(f"Unsupported image ndim: {rescaled_image.ndim}. Image must be 2D or 3D.")

        unscaled_image = resize(
            rescaled_image,
            output_shape,
            order=order,
            mode='constant',
            cval=0,
            anti_aliasing=anti_aliasing,
            preserve_range=True
        ).astype(rescaled_image.dtype)
        return unscaled_image
