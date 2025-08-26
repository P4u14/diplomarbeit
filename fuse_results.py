import os
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# Configuration
results_to_fuse = [
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
GT_PATH = "data/Validation_Data_Small"
OUTPUT_DIR = "data/Fused_Results/"

# Farben für die Umrandung
PINK = (255, 105, 180, 255 )   # Helleres Neon-Pink
ORANGE = (255, 165, 0, 255)   # Helleres Neon-Orange


def find_contours(mask_arr, thickness=1):
    from scipy.ndimage import binary_dilation
    mask = mask_arr > 0
    dilated = mask.copy()
    for _ in range(thickness):
        dilated = binary_dilation(dilated)
    contour = np.logical_and(dilated, ~mask)
    return contour

def overlay_contour(base_img, mask_img, color, thickness=1):
    mask_arr = np.array(mask_img.convert("L"))
    contour = find_contours(mask_arr, thickness=thickness)
    overlay = Image.new("RGBA", base_img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    ys, xs = np.where(contour)
    for x, y in zip(xs, ys):
        draw.point((x, y), fill=color)
    return Image.alpha_composite(base_img.convert("RGBA"), overlay)

def process_image(img_name, fuse_dir, gt_path, output_dir):
    # Masks
    seg_mask_path = os.path.join(fuse_dir, img_name)
    gt_mask_path = os.path.join(gt_path, img_name)

    # Original image
    orig_path = os.path.join(gt_path, img_name.replace('-mask.Gauss.png', '.Gauss.png'))
    if not os.path.exists(orig_path):
        print('Original image not found for', img_name)
        return

    base_img = Image.open(orig_path)

    result = base_img.copy()
    if os.path.exists(seg_mask_path):
        seg_mask = Image.open(seg_mask_path)
        result = overlay_contour(result, seg_mask, ORANGE, thickness=2)
    if os.path.exists(gt_mask_path):
        gt_mask = Image.open(gt_mask_path)
        result = overlay_contour(result, gt_mask, PINK, thickness=2)
    out_path = os.path.join(output_dir, img_name)
    result.save(out_path)

def main():
    for fuse_dir in results_to_fuse:
        output_path = OUTPUT_DIR +"/" + os.path.basename(fuse_dir)
        os.makedirs(output_path, exist_ok=True)
        for img_name in tqdm(os.listdir(fuse_dir), desc='Preprocessing image for ' + os.path.basename(fuse_dir)):
            if not img_name.endswith('.png') or '-mask' not in img_name:
                continue
            process_image(img_name, fuse_dir, GT_PATH, output_path)

if __name__ == "__main__":
    main()
