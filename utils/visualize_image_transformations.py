#!/usr/bin/env python3
"""
Script to visualize image transformations by applying one or more preprocessing steps.
Usage:
    python visualize_image_transformations.py \
        --image /path/to/image.png \
        --steps preprocessing.square_image_preprocessor.SquareImagePreprocessor \
                preprocessing.blue_color_preprocessor.BlueColorPreprocessor \
        --output_dir /path/to/output
"""
import os
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    raise ImportError("Please install imageio (pip install imageio) to use this script.")

# --- Konfiguration ---
# Pfade
IMAGE_PATH = '~/Documents/DA/PreprocessingVis/gkge_61_1_635226079315283367.Gauss.png'
# Optional: Pfad zur Segmentierungs-Maske (für Refinement-Schritte)
SEGMENTATION_PATH = '~/Documents/DA/PreprocessingVis/gkge_61_1_635226079315283367-mask.Gauss.png'
OUTPUT_DIR = '~/Documents/DA/PreprocessingVis/'
# Schritte
# Preprocessing-Instanzen: Klassen mit preprocess_image(img) -> (arr, params)
from preprocessing.blue_color_preprocessor import BlueColorPreprocessor
PREPROCESSORS = [
    # SquareImagePreprocessor(),
    # BlueColorPreprocessor()
    # TorsoRoiPreprocessor(target_ratio=5 / 7)
    # DimplesRoiPreprocessor(target_ratio=10 / 7)
]
# Refinement-Instanzen: Klassen mit refine(seg, target_image)
from postprocessing.color_patch_refiner import ColorPatchRefiner
from target_image.target_image import TargetImage
REFINERS = [
    ColorPatchRefiner(BlueColorPreprocessor()),
]

def load_image(path):
    return imageio.imread(path)

def save_image(path, arr):
    imageio.imwrite(path, arr)

def main():
    # Pfade und Ausgabeverzeichnis
    image_path = IMAGE_PATH
    seg_path = SEGMENTATION_PATH or None
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    # Bild laden
    image = load_image(image_path)
    # Ziel-Objekt für Refinement
    target = TargetImage(image_path)
    # Wenn Segmentierungspfad gesetzt, laden
    segmentation = None
    if seg_path:
        segmentation = load_image(seg_path)
    # 1) Preprocessing-Schritte
    for proc in PREPROCESSORS:
        name = proc.__class__.__name__
        try:
            processed, params = proc.preprocess_image(image)
        except Exception as e:
            print(f"Error in preprocessing '{name}': {e}")
            continue
        # speichern
        arr = processed
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        elif arr.dtype != np.uint8:
            # assume float in [0,1]
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)

        # Handle channel dimensions
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        filename = f"pre_{name}.png"
        path = os.path.join(output_dir, filename)
        save_image(path, arr)
        print(f"Saved preprocessing {name} to {path}")
    # 2) Refinement-Schritte (benötigen Segmentierung)
    if segmentation is not None:
        for ref in REFINERS:
            name = ref.__class__.__name__
            try:
                refined = ref.refine(segmentation, target)
            except Exception as e:
                print(f"Error in refinement '{name}': {e}")
                continue
            arr = refined
            # normalisieren wie oben
            if arr.dtype == bool:
                arr = arr.astype(np.uint8) * 255
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            filename = f"ref_{name}.png"
            path = os.path.join(output_dir, filename)
            save_image(path, arr)
            print(f"Saved refinement {name} to {path}")
    else:
        print("No segmentation path provided; skipping refinement.")

if __name__ == '__main__':
    main()
