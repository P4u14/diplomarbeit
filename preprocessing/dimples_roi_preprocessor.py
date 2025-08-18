from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor


class DimplesRoiPreprocessor(TorsoRoiPreprocessor):

    def __init__(self, target_ratio):
        super().__init__(target_ratio)

    def preprocess(self, image):
        original_size = {
            'height': image.shape[0],
            'width': image.shape[1]
        }
        # show_image(image, "0. Original image", True)
        cropped_image, bbox = self.crop_dimples_roi(image)
        # show_image(cropped_image, "1. Preprocess: Cropped Dimples ROI", True)
        cropped_and_padded_image, padding, padded_size = super().pad_image_to_correct_ratio(cropped_image, bbox)
        # show_image(cropped_and_padded_image, "2. Preprocess: Padded Image", True)
        resized_image = super().rescale_image(cropped_and_padded_image, original_size)
        # show_image(resized_image, "3. Preprocess: Resized Image (Final)", True)
        parameters = {
            'original_size': original_size,
            'bbox': bbox,
            'padding': padding,
            'padded_size': padded_size,
        }
        return resized_image, parameters

    def preprocess_with_parameters(self, image, parameters):
         return super().preprocess_with_parameters(image, parameters)

    def undo_preprocessing(self, preprocessed_image, parameters, is_already_color=False):
        return super().undo_preprocessing(preprocessed_image, parameters, is_already_color)

    def crop_dimples_roi(self, image):
        torso_roi_image, torso_bbox = super().crop_torso_roi(image)
        torso_roi_height, torso_roi_width = torso_roi_image.shape[:2]
        # show_image(torso_roi_image, "0.5. Preprocess: Cropped Torso ROI", True)
        dimples_bbox_from_torso = {
            'min_x': 0,
            'max_x': torso_roi_width,
            'min_y': torso_roi_height // 2,
            'max_y': torso_roi_height
        }
        dimples_roi_image = torso_roi_image[dimples_bbox_from_torso['min_y']:dimples_bbox_from_torso['max_y'], dimples_bbox_from_torso['min_x']:dimples_bbox_from_torso['max_x']]
        dimples_bbox_from_original = {
            'min_x': torso_bbox['min_x'],
            'max_x': torso_bbox['min_x'] + torso_roi_width,
            'min_y': torso_bbox['min_y'] + torso_roi_height // 2,
            'max_y': torso_bbox['min_y'] + torso_roi_height,
        }
        return dimples_roi_image, dimples_bbox_from_original
