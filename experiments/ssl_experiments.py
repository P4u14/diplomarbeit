from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from preprocessing.square_image_preprocessor import SquareImagePreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor
from segmenter.ml_segmenter import MLSegmenter

SSL_BASE_OUTPUT_DIR = "data/Results/Segmentation_Results/SSL/"

ssl_experiments = {
    1: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/jigsaw_abs_pos/attention_unet/Experiment01-test/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=ColorPatchRefiner(color_preprocessor=BlueColorPreprocessor()),
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment01"
    ),
}