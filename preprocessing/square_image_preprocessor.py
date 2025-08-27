from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
from preprocessing.preprocessing_step import IPreprocessingStep
# from preprocessing.torso_roi_preprocessor import show_image


class SquareImagePreprocessor(IPreprocessingStep):

    def __init__(self, resize_size: int = 256, crop_size: int = 255):
        assert resize_size >= crop_size, "resize_size must be >= crop_size"
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)

    def preprocess_image(self, image):
        H, W = image.shape[:2]
        pil, mode = self._to_pil(image)

        # 1) Resize (bilinear)
        pil = pil.resize((self.resize_size, self.resize_size), resample=Image.BILINEAR)

        # 2) Center crop
        dh = self.resize_size - self.crop_size
        dw = self.resize_size - self.crop_size
        # round off left/top (like torchvision)
        left   = int(np.floor(dw / 2.0))
        top    = int(np.floor(dh / 2.0))
        right  = left + self.crop_size
        bottom = top + self.crop_size
        pil = pil.crop((left, top, right, bottom))

        out = self._from_pil(pil, mode, image.ndim)

        params = {
            "orig_size": (H, W),
            "resize_size": self.resize_size,
            "crop_box": (left, top, right, bottom),
            "out_mode": mode,
            "orig_ndim": image.ndim,
        }
        return out, params

    def preprocess_mask(self, mask: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply the same steps to a mask:
        - Resize with NEAREST
        - CenterCrop with the same crop_box
        Expects 2D or 3D (H,W,1) mask.
        """
        assert mask.ndim in (2, 3), f"Unexpected mask shape: {mask.shape}"
        # For masks, values usually remain {0.1} / {0.255}; never use bilinear blending.
        pil_m, _ = self._to_pil(mask.astype(np.uint8))  # auf uint8 für PIL

        # Resize (nearest)
        pil_m = pil_m.resize((self.resize_size, self.resize_size), resample=Image.NEAREST)

        # Crop with the same crop box as for the image
        if params is None:
            dh = self.resize_size - self.crop_size
            dw = self.resize_size - self.crop_size
            left = int(np.floor(dw / 2.0))
            top = int(np.floor(dh / 2.0))
            right = left + self.crop_size
            bottom = top + self.crop_size
        else:
            left, top, right, bottom = params["crop_box"]

        pil_m = pil_m.crop((left, top, right, bottom))

        arr = np.array(pil_m)
        if mask.ndim == 3:
            arr = arr[:, :, None]
        return arr

    def undo_preprocessing(self, mask: np.ndarray, params: Dict[str, Any], is_already_color=False) -> np.ndarray:
        """
        Save the steps for a mask:
        1) Place the mask (crop_size x crop_size) in a (resize_size x resize_size) canvas
        back to the same crop_box (rest = 0).
        2) Scale the canvas back to orig_size with NEAREST.
        """
        resize_size = int(params["resize_size"])
        left, top, right, bottom = params["crop_box"]
        orig_h, orig_w = params["orig_size"]

        # Step 1: Back to Resize Canvas
        if mask.ndim == 3:
            canvas = np.zeros((resize_size, resize_size, mask.shape[2]), dtype=mask.dtype)
            canvas[top:bottom, left:right, :] = mask
        else:
            canvas = np.zeros((resize_size, resize_size), dtype=mask.dtype)
            canvas[top:bottom, left:right] = mask

        # Step 2: NEAREST back to original size
        pil = Image.fromarray(canvas.squeeze() if canvas.ndim == 3 and canvas.shape[2] == 1 else canvas)
        pil = pil.resize((orig_w, orig_h), resample=Image.NEAREST)
        out = np.array(pil)

        if mask.ndim == 3 and out.ndim == 2:
            out = out[:, :, None]
        # show_image(out, "3. Undo: Un-cropped (Final)", is_already_color)
        return out

    @staticmethod
    def _to_pil(img: np.ndarray) -> Tuple[Image.Image, str]:
        """
        Takes np.ndarray with shape (H,W) or (H,W,C), dtype uint8/float.
        Returns PIL image and a hint on how to scale when converting back.
        """
        assert img.ndim in (2, 3), f"Unexpected image shape: {img.shape}"
        mode = "uint8"
        if img.dtype != np.uint8:
            # accept float; in [0,1] → scale to uint8
            mode = "float01"
            arr = np.clip(img, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = img

        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L")
        else:
            if arr.shape[2] == 1:
                pil = Image.fromarray(arr.squeeze(2), mode="L")
            elif arr.shape[2] == 3:
                pil = Image.fromarray(arr, mode="RGB")
            else:
                # RGBA -> RGB
                pil = Image.fromarray(arr[:, :, :3], mode="RGB")
        return pil, mode

    @staticmethod
    def _from_pil(pil: Image.Image, out_mode: str, orig_ndim: int) -> np.ndarray:
        """
        Converts PIL back to np.ndarray. out_mode==“float01” → scales to [0,1] float32,
        otherwise uint8. orig_ndim controls whether (H,W) or (H,W,1/3) is returned.
        """
        arr = np.array(pil)
        if arr.ndim == 2 and orig_ndim == 3:
            arr = arr[:, :, None]  # (H,W,1)

        if out_mode == "float01":
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.uint8)
        return arr

