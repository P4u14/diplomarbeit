from validation.validator import Validator

segmentations_to_validate = [
    # "data/Atlas_Experiment01",
    # "data/Atlas_Experiment02",
    # "data/Atlas_Experiment03",
    # "data/Atlas_Experiment04",
    # "data/Atlas_Experiment05",
    # "data/Atlas_Experiment06",
    # "data/Atlas_Experiment07",
    # "data/Atlas_Experiment08",
    # "data/Atlas_Experiment09",
    # "data/Atlas_Experiment10",
    # "data/Atlas_Experiment11",
    # "data/Atlas_Experiment12",
    # "data/Atlas_Experiment13",
    # "data/Atlas_Experiment14",
    # "data/Atlas_Experiment15",
    # "data/Atlas_Experiment16",
    # "data/Atlas_Experiment17",
    # "data/Atlas_Experiment18",
    # "data/Atlas_Experiment19",
    # "data/Atlas_Experiment20",
    # "data/Atlas_Experiment21",
    # "data/Atlas_Experiment22",
    # "data/Atlas_Experiment23",
    # "data/Atlas_Experiment24",
    # "data/Atlas_Experiment25",
    # "data/Atlas_Experiment26",
    # "data/Atlas_Experiment27",
    # "data/Atlas_Experiment28",
    # "data/Atlas_Experiment29",
    # "data/Atlas_Experiment30",
    # "data/Atlas_Experiment31",
    # "data/Atlas_Experiment32",
    # "data/Atlas_Experiment33",
    # "data/Atlas_Experiment34",
    # "data/Atlas_Experiment35",
    # "data/Atlas_Experiment36",
    # "data/Atlas_Experiment37",
    # "data/Atlas_Experiment38",
    # "data/Atlas_Experiment39",
    # "data/Atlas_Experiment40",
    "data/Atlas_Experiment41"
]
OUTPUT_DIR = "data/Validation_Results/"

if __name__ == "__main__":
    validator = Validator(ground_truth_dir="data/Validation_Data_Small", output_dir=OUTPUT_DIR)

    for segmentations_dir in segmentations_to_validate:
        validator.validate(predictions_dir=segmentations_dir)
