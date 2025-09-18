from validation.validator import Validator

segmentations_to_validate = [
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment37",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment38",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment39",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment40",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment41",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment78",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment79",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment80",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment81",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment82",
]
OUTPUT_DIR = "data/Results/Validation/"

if __name__ == "__main__":
    validator = Validator(ground_truth_dir="data/Images/Validation_Data_Small", output_dir=OUTPUT_DIR)

    for segmentations_dir in segmentations_to_validate:
        validator.validate(predictions_dir=segmentations_dir)
