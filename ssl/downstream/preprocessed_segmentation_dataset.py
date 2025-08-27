from typing import Sequence, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _pil_to_np(img: Image.Image) -> np.ndarray:
    """PIL -> np.uint8, Shape (H,W) oder (H,W,3). RGBA -> RGB."""
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _np_to_tensor_chw01(arr: np.ndarray) -> torch.Tensor:
    """
    np.uint8/float -> torch.float32 in [0,1], Shape (C,H,W).
    Accepts (H,W), (H,W,1), (H,W,3).
    """
    if arr.ndim == 2:
        arr = arr[:, :, None]          # (H,W)->(H,W,1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # (H,W,C)->(C,H,W)
    return torch.from_numpy(arr)


class PreprocessedSegmentationDataset(Dataset):
    """
    Loads images + masks and applies a sequence of IPreprocessingStep instances.
    Each step must have the following methods:
      - preprocess_image(image) -> (image_out, params)
      - preprocess_mask(mask, params) -> mask_out
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        preprocessing_steps: Optional[Sequence] = None,
    ):
        assert len(image_paths) == len(mask_paths), "image_paths and mask_paths need to have the same length."
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.preprocessing_steps = list(preprocessing_steps or [])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load
        img  = Image.open(img_path).convert("RGB")     # training expects RGB
        mask = Image.open(mask_path)                   # mask as saved (L/PNG)
        img_np  = _pil_to_np(img)                      # (H,W,3) uint8
        mask_np = np.array(mask)                       # (H,W) or (H,W,3/1)

        # Pipeline: same steps on image+mask (with params from image)
        params_chain = []
        for step in self.preprocessing_steps:
            img_np, params = step.preprocess_image(img_np)
            mask_np = step.preprocess_mask(mask_np, params)
            params_chain.append(params)

        # If mask has more than 2 dimensions, reduce to first channel
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        # Binarization
        if mask_np.max() > 1.0:
            mask_np = (mask_np > 127).astype(np.float32)
        else:
            mask_np = (mask_np > 0).astype(np.float32)

        # Convert mask to tensor (1, H, W)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)

        # Convert image to tensor (3,H,W), float32 [0,1]
        img_t  = _np_to_tensor_chw01(img_np)

        return img_t, mask_t, img_path, mask_path
