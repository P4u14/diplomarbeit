from validation.validator import Validator

segmentations_to_validate = [
    "data/Segmentation_Results/Atlas_Experiment01",
    "data/Segmentation_Results/Atlas_Experiment02",
    "data/Segmentation_Results/Atlas_Experiment03",
    "data/Segmentation_Results/Atlas_Experiment04",
    "data/Segmentation_Results/Atlas_Experiment05",
    "data/Segmentation_Results/Atlas_Experiment06",
    "data/Segmentation_Results/Atlas_Experiment07",
    "data/Segmentation_Results/Atlas_Experiment08",
    "data/Segmentation_Results/Atlas_Experiment09",
    "data/Segmentation_Results/Atlas_Experiment10",
    "data/Segmentation_Results/Atlas_Experiment11",
    "data/Segmentation_Results/Atlas_Experiment12",
    "data/Segmentation_Results/Atlas_Experiment13",
    "data/Segmentation_Results/Atlas_Experiment14",
    "data/Segmentation_Results/Atlas_Experiment15",
    "data/Segmentation_Results/Atlas_Experiment16",
    "data/Segmentation_Results/Atlas_Experiment17",
    "data/Segmentation_Results/Atlas_Experiment18",
    "data/Segmentation_Results/Atlas_Experiment19",
    "data/Segmentation_Results/Atlas_Experiment20",
    "data/Segmentation_Results/Atlas_Experiment21",
    "data/Segmentation_Results/Atlas_Experiment22",
    "data/Segmentation_Results/Atlas_Experiment23",
    "data/Segmentation_Results/Atlas_Experiment24",
    "data/Segmentation_Results/Atlas_Experiment25",
    "data/Segmentation_Results/Atlas_Experiment26",
    "data/Segmentation_Results/Atlas_Experiment27",
    "data/Segmentation_Results/Atlas_Experiment28",
    "data/Segmentation_Results/Atlas_Experiment29",
    "data/Segmentation_Results/Atlas_Experiment30",
    "data/Segmentation_Results/Atlas_Experiment31",
    "data/Segmentation_Results/Atlas_Experiment32",
    "data/Segmentation_Results/Atlas_Experiment33",
    "data/Segmentation_Results/Atlas_Experiment34",
    "data/Segmentation_Results/Atlas_Experiment35",
    "data/Segmentation_Results/Atlas_Experiment36",
    "data/Segmentation_Results/Atlas_Experiment37",
    "data/Segmentation_Results/Atlas_Experiment38",
    "data/Segmentation_Results/Atlas_Experiment39",
    "data/Segmentation_Results/Atlas_Experiment40",
    "data/Segmentation_Results/Atlas_Experiment41",
    "data/Segmentation_Results/Atlas_Experiment42",
]
OUTPUT_DIR = "data/Validation_Results/"

if __name__ == "__main__":
    validator = Validator(ground_truth_dir="data/Validation_Data_Small", output_dir=OUTPUT_DIR)

    for segmentations_dir in segmentations_to_validate:
        validator.validate(predictions_dir=segmentations_dir)
