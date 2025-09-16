from atlas.refiner.color_patch_refiner import ColorPatchRefiner
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
from preprocessing.square_image_preprocessor import SquareImagePreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor
from segmenter.ml_segmenter import MLSegmenter

SSL_BASE_OUTPUT_DIR = "data/Results/Segmentation_Results/SSL/"

ssl_experiments = {
    1: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment01/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment01"
    ),
    2: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment02/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment02"
    ),
    3: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment03/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment03"
    ),
    4: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment04/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment04"
    ),
    5: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment05/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment05"
    ),
    6: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment06/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment06"
    ),
    7: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment07/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment07"
    ),
    8: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment08/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment08"
    ),
    9: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment09/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment09"
    ),
    10: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment10/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment10"
    ),
    11: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment11/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=25,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment11"
    ),
    12: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment12/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=25,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment12"
    ),

    13: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment13/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment13"
    ),
    14: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment14/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment14"
    ),
    15: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment15/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment15"
    ),
    16: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment16/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment16"
    ),
    17: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment17/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment17"
    ),
    18: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment18/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment18"
    ),
    19: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment19/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment19"
    ),
    20: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment20/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment20"
    ),
    21: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment21/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment21"
    ),
    22: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment22/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment22"
    ),
    # 23: MLSegmenter(
    #     model_type="attention_unet",
    #     weights_path="data/SSL_Downstream/Experiment23/best_model.pth",
    #     backbone="jigsaw_abs_pos",
    #     pretext_classes=25,
    #     downstream_classes=1,
    #     device=None,
    #     preprocessing_steps=[SquareImagePreprocessor()],
    #     segmentation_refiner=None,
    #     output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment23"
    # ),
    # 24: MLSegmenter(
    #     model_type="attention_unet",
    #     weights_path="data/SSL_Downstream/Experiment24/best_model.pth",
    #     backbone="jigsaw_abs_pos",
    #     pretext_classes=25,
    #     downstream_classes=1,
    #     device=None,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
    #     segmentation_refiner=None,
    #     output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment24"
    # ),

    25: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment25/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment25"
    ),
    26: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment26/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment26"
    ),
    27: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment27/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment27"
    ),
    28: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment28/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment28"
    ),
    29: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment29/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment29"
    ),
    30: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment30/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment30"
    ),
    31: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment31/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment31"
    ),
    32: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment32/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment32"
    ),
    33: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment33/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment33"
    ),
    34: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment34/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment34"
    ),
    35: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment35/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=25,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment35"
    ),
    36: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment36/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=25,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment36"
    ),

    37: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment37/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment37"
    ),
    38: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment38/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment38"
    ),
    39: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment39/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment39"
    ),
    40: MLSegmenter(
        model_type="segmentation_head",
        weights_path="data/SSL_Downstream/Experiment40/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment40"
    ),
    41: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment41/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment41"
    ),
    42: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment42/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=100,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment42"
    ),
    43: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment43/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment43"
    ),
    44: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment44/best_model.pth",
        backbone="jigsaw_perm",
        pretext_classes=1000,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment44"
    ),
    45: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment45/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment45"
    ),
    46: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment46/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=9,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment46"
    ),
    47: MLSegmenter(
        model_type="attention_unet",
        weights_path="data/SSL_Downstream/Experiment47/best_model.pth",
        backbone="jigsaw_abs_pos",
        pretext_classes=25,
        downstream_classes=1,
        device=None,
        preprocessing_steps=[SquareImagePreprocessor()],
        segmentation_refiner=None,
        output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment47"
    ),
    # 48: MLSegmenter(
    #     model_type="attention_unet",
    #     weights_path="data/SSL_Downstream/Experiment48/best_model.pth",
    #     backbone="jigsaw_abs_pos",
    #     pretext_classes=25,
    #     downstream_classes=1,
    #     device=None,
    #     preprocessing_steps=[TorsoRoiPreprocessor(target_ratio=5 / 7), SquareImagePreprocessor()],
    #     segmentation_refiner=None,
    #     output_dir=SSL_BASE_OUTPUT_DIR + "SSL_Experiment48"
    # ),
}