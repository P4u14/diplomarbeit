from validation.validator import Validator

segmentations_to_validate = [
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment01",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment02",
    "data/Results/Segmentation_Results/Atlas/Atlas_Experiment03",
]
OUTPUT_DIR = "data/Results/Validation/"

if __name__ == "__main__":
    validator = Validator(ground_truth_dir="data/Images/Validation_Data_Small", output_dir=OUTPUT_DIR)

    for segmentations_dir in segmentations_to_validate:
        validator.validate(predictions_dir=segmentations_dir)
