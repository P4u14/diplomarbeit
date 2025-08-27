import os
from typing import Optional, Sequence

import torch
from PIL.Image import Image
from torch.utils.data import Dataset

from ssl.downstream.preprocessed_segmentation_dataset import _pil_to_np, _np_to_tensor_chw01


class PreprocessedImageDataset(Dataset):
    """
    Loads (image_path, label) from a .txt list such as ilsvrc12_train.txt
    and applies preprocessing_steps to the image.
    File list format: “<rel_path> <label>”
    """
    def __init__(self, root_dir: str, list_txt: str, preprocessing_steps: Optional[Sequence]=None):
        self.root_dir = root_dir.rstrip("/")
        self.preprocessing_steps = list(preprocessing_steps or [])
        self.items = []
        with open(list_txt, "r") as f:
            for line in f:
                name, label = line.strip().split()
                self.items.append((name, int(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rel, label = self.items[idx]
        path = os.path.join(self.root_dir, rel)
        img = Image.open(path).convert("RGB")
        img_np = _pil_to_np(img)

        for step in self.preprocessing_steps:
            img_np, _ = step.preprocess_image(img_np)

        img_t = _np_to_tensor_chw01(img_np)  # (3,H,W) float32 [0,1]
        return img_t, torch.tensor(label, dtype=torch.long)