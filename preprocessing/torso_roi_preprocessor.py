import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from preprocessing.preprocessing_step import IPreprocessingStep


def show_image(image, title="", is_already_color=False):
    """Helper function to display an image."""
    if not is_already_color:
        display_image = (image * 255).astype(np.uint8)
    else:
        display_image = image
    h, w = image.shape[:2]

    plt.imshow(display_image)
    plt.title(f"{title} ({h}x{w})")
    plt.axis('off')
    plt.show()


class TorsoRoiPreprocessor(IPreprocessingStep):

    def __init__(self, target_ratio):
        self.target_ratio = target_ratio # width : height

    def preprocess(self, image):
        original_size = {
            'height': image.shape[0],
            'width': image.shape[1]
        }
        cropped_image, bbox = self.crop_torso_roi(image)
        # show_image(cropped_image, "1. Preprocess: Cropped Torso ROI", True)
        cropped_and_padded_image, padding, padded_size = self.pad_image_to_correct_ratio(cropped_image, bbox)
        # show_image(cropped_and_padded_image, "2. Preprocess: Padded Image", True)
        resized_image = self.rescale_image(cropped_and_padded_image, original_size)
        # show_image(resized_image, "3. Preprocess: Resized Image (Final)", True)
        parameters = {
            'original_size': original_size,
            'bbox': bbox,
            'padding': padding,
            'padded_size': padded_size,
        }
        return resized_image, parameters

    def preprocess_with_parameters(self, image, parameters):
        cropped_image = self.crop_with_parameters(image, parameters['bbox'])
        # show_image(cropped_image, "1. Preprocess w/ Params: Cropped")
        cropped_and_padded_image = self.pad_image_with_parameters(cropped_image, parameters['padding'])
        # show_image(cropped_and_padded_image, "2. Preprocess w/ Params: Padded")
        resized_image = self.rescale_image(cropped_and_padded_image, parameters['original_size'])
        # show_image(resized_image, "3. Preprocess w/ Params: Resized (Final)")
        return resized_image

    def undo_preprocessing(self, preprocessed_image, parameters, is_already_color=False):
        show_image(preprocessed_image, "0. Preprocessed Image", is_already_color)
        cropped_and_padded_image = self.undo_rescale_image(preprocessed_image, parameters['padded_size'])
        show_image(cropped_and_padded_image, "1. Undo: Un-rescaled", is_already_color)
        cropped_image = self.undo_pad_image_to_correct_ratio(cropped_and_padded_image, parameters['padding'])
        show_image(cropped_image, "2. Undo: Un-padded", is_already_color)
        image = self.undo_crop_torso_roi(cropped_image, parameters['original_size'], parameters['bbox'])
        show_image(image, "3. Undo: Un-cropped (Final)", is_already_color)
        return image

    @staticmethod
    def crop_torso_roi(image):
        image_rgb = image[..., :3]
        torso_roi = np.any(image_rgb != [0, 0, 0], axis=-1)
        coords = np.argwhere(torso_roi)
        if coords.size == 0:
            raise ValueError("No torso region found in the image.")
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        bbox = {
            'min_x': min_x,
            'max_x': max_x + 1, # use exclusive values
            'min_y': min_y,
            'max_y': max_y + 1# use exclusive values
        }
        cropped_image = image[min_y:max_y, min_x:max_x, ...]
        return cropped_image, bbox

    @staticmethod
    def crop_with_parameters(image, bbox):
        cropped_image = image[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], ...]
        return cropped_image

    @staticmethod
    def undo_crop_torso_roi(cropped_image, original_size, bbox):
        restored_image = np.zeros((original_size['height'], original_size['width'], cropped_image.shape[2]), dtype=cropped_image.dtype)
        if restored_image.shape[2] == 4:
            restored_image[..., 3] = 255
        restored_image[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x']] = cropped_image
        return restored_image

    def pad_image_to_correct_ratio(self, cropped_image, bbox):
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
        padded_image = np.pad(
            cropped_image,
            ((padding['top'], padding['bottom']), (padding['left'], padding['right']), (0, 0)),
            mode='constant',
            constant_values=0
        )
        return padded_image

    @staticmethod
    def undo_pad_image_to_correct_ratio(padded_image, padding):
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
