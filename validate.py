from validation.validator import Validator

segmentations_to_validate = [
    "data/Atlas_Experiment01",
    "data/Atlas_Experiment02",
    "data/Atlas_Experiment23",
]
OUTPUT_DIR = "data/Validation_Results/"

if __name__ == "__main__":
    validator = Validator(ground_truth_dir="data/Validation_Data_Small", output_dir=OUTPUT_DIR)

    for segmentations_dir in segmentations_to_validate:
        validator.validate(predictions_dir=segmentations_dir)
