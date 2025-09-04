from experiments.atlas_experiments import atlas_experiments
from segmenter.segmentation_runner import SegmentationRunner

ATLAS_DIR = "data/Images/Atlas_Data"
ATLAS_DIR_BMI_PERCENTILE = "data/Images/Atlas_Data_BMI_Percentile"
TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"
IMAGE_INFO_PATH = "data/Info_Sheets/All_Data_Renamed_overview.csv"
BMI_TABLE_PATH = "data/Info_Sheets/bmi_table_who.csv"

image_segmenter = [
    atlas_experiments[12],
    atlas_experiments[22],
    atlas_experiments[36],
    atlas_experiments[42],
    atlas_experiments[43],
    atlas_experiments[44],
    atlas_experiments[45],
    atlas_experiments[46],
    atlas_experiments[47],
    atlas_experiments[48],
    atlas_experiments[49],
    atlas_experiments[50],
    atlas_experiments[51],
    atlas_experiments[52],
    atlas_experiments[53],
    atlas_experiments[54],
    atlas_experiments[55],
    atlas_experiments[56],
    atlas_experiments[57],
    atlas_experiments[58],
    atlas_experiments[59],
    atlas_experiments[60],
    atlas_experiments[61],
    atlas_experiments[62],
    atlas_experiments[63],
    atlas_experiments[64],
    atlas_experiments[65],
    atlas_experiments[66],
    atlas_experiments[67],
    atlas_experiments[68],
    atlas_experiments[69],
    atlas_experiments[70],
    atlas_experiments[71],
    atlas_experiments[72],
    atlas_experiments[73],
    atlas_experiments[74],
    atlas_experiments[75],
    atlas_experiments[76],
    atlas_experiments[77],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        SegmentationRunner(segmenter, TARGET_IMAGES_DIR).run()
