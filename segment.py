from experiments.atlas_experiments import atlas_experiments
from segmenter.segmentation_runner import SegmentationRunner

ATLAS_DIR = "data/Images/Atlas_Data"
ATLAS_DIR_BMI_PERCENTILE = "data/Images/Atlas_Data_BMI_Percentile"
TARGET_IMAGES_DIR = "data/Validation_Data_Small"
IMAGE_INFO_PATH = "data/Info_Sheets/All_Data_Renamed_overview.csv"
BMI_TABLE_PATH = "data/Info_Sheets/bmi_table_who.csv"

image_segmenter = [
    atlas_experiments[1],
    atlas_experiments[2],
    atlas_experiments[3],
    atlas_experiments[4],
    atlas_experiments[5],
    atlas_experiments[6],
    atlas_experiments[7],
    atlas_experiments[8],
    atlas_experiments[9],
    atlas_experiments[10],
    atlas_experiments[11],
    atlas_experiments[12],
    atlas_experiments[13],
    atlas_experiments[14],
    atlas_experiments[15],
    atlas_experiments[16],
    atlas_experiments[17],
    atlas_experiments[18],
    atlas_experiments[19],
    atlas_experiments[20],
    atlas_experiments[21],
    atlas_experiments[22],
    atlas_experiments[23],
    atlas_experiments[24],
    atlas_experiments[25],
    atlas_experiments[26],
    atlas_experiments[27],
    atlas_experiments[28],
    atlas_experiments[29],
    atlas_experiments[30],
    atlas_experiments[31],
    atlas_experiments[32],
    atlas_experiments[33],
    atlas_experiments[34],
    atlas_experiments[35],
    atlas_experiments[36],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        SegmentationRunner(segmenter, TARGET_IMAGES_DIR).run()
