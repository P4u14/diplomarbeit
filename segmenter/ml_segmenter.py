import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from segmenter.base_segmenter import BaseSegmenter
from ssl.downstream.simple_classifier.segmentation_head import SegmentationHead
from ssl.downstream.unet.jigsaw_absolut_position.unet_decoder_jigsaw_abs_pos import UnetDecoderJigsawAbsPos
from ssl.downstream.unet.jigsaw_permutation.alex_net_attention_unet_decoder import AlexNetAttentionUNetDecoder
from ssl.pretext.jigsaw_permutation.JigsawPermNetwork import JigsawPermNetwork
from target_image.target_image import TargetImage
from target_image.target_segmentation import TargetSegmentation


class MLSegmenter(BaseSegmenter):

    def __init__(self, model_type, weights_path, backbone, pretext_classes, downstream_classes, output_dir, device=None, preprocessing_steps=None, segmentation_refiner=None):
        super().__init__(output_dir, preprocessing_steps, segmentation_refiner)
        self.model_type = model_type
        self.weights_path = weights_path
        self.backbone = backbone
        self.pretext_classes = pretext_classes
        self.downstream_classes = downstream_classes
        self.device = self.select_device(device)
        self.model = self.build_model().to(self.device)
        self.load_model()

    @staticmethod
    def select_device(selected_device=None):
        if selected_device is not None:
            return torch.device(selected_device)
        if torch.backends.mps.is_available(): # Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")

    def build_model(self):
        if self.model_type == "attention_unet" and self.backbone == "jigsaw_abs_pos":
            return UnetDecoderJigsawAbsPos(pretext_model_path=None, num_classes=self.downstream_classes, pretext_classes=self.pretext_classes)
        elif self.model_type == "attention_unet" and self.backbone == "jigsaw_perm":
            encoder = JigsawPermNetwork(classes=self.pretext_classes).conv
            return AlexNetAttentionUNetDecoder(encoder=encoder, num_classes=self.downstream_classes)
        elif self.model_type == "segmentation_head" and self.backbone == "jigsaw_perm":
            encoder = JigsawPermNetwork(classes=self.pretext_classes).conv
            head = SegmentationHead(num_classes=self.downstream_classes)
            return nn.Sequential(encoder, head)
        else:
            raise ValueError(f"Unknown combination of model_type: {self.model_type} and backbone: {self.backbone}")

    def load_model(self):
        ckpt = torch.load(self.weights_path, map_location=self.device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded: {self.weights_path}")

    @staticmethod
    def _ensure_tensor_chw(arr_or_tensor) -> torch.Tensor:
        if torch.is_tensor(arr_or_tensor):
            t = arr_or_tensor
            if t.dim() == 2:
                t = t.unsqueeze(0)  # (1,H,W)
            elif t.dim() == 3 and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            # otherwise already (C,H,W)
            t = t.float()
            if t.max() > 1.0:
                t = t / 255.0
            return t

        # numpy / list
        arr = np.asarray(arr_or_tensor)
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1,H,W)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            pass
        else:
            raise ValueError("Unexpected image shape; need (H,W), (H,W,C) or (C,H,W)")
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return torch.from_numpy(arr)

    @torch.inference_mode()
    def segment_images(self, target_images: list[TargetImage]):
        for target_image in tqdm(target_images, desc="Segmenting images (1-by-1)"):
            # Preprocessing
            for pp_step in self.preprocessing_steps:
                target_image.preprocessed_image, parameters = pp_step.preprocess_image(target_image.preprocessed_image)
                target_image.append_preprocessing_parameters(parameters)

            # To tensor, on device, batch dim
            x = self._ensure_tensor_chw(target_image.preprocessed_image).unsqueeze(0).to(self.device)  # (1,C,H,W)
            in_h, in_w = x.shape[-2:]

            # Forward, map logits to input size if necessary (as in training)
            logits = self.model(x)  # (1,1,h,w) oder (1,C,h,w)
            if logits.shape[-2:] != (in_h, in_w):
                logits = F.interpolate(logits, size=(in_h, in_w), mode='bilinear', align_corners=False)

            # Binary target_mask (1 downstream class)
            if self.downstream_classes == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()  # (1,1,H,W)
                target_mask = pred.squeeze(0).squeeze(0).cpu().numpy()  # (H,W) in {0,1}
            else:
                # If multiple classes: Softmax + Argmax, etc.
                probs = torch.softmax(logits, dim=1)
                cls = probs.argmax(dim=1).squeeze(0).cpu().numpy()  # (H,W) in {0..C-1}
                target_mask = (cls > 0).astype(np.float32)

            # Undo-Preprocessing (reversed))
            for pp_step, params in reversed(list(zip(self.preprocessing_steps, target_image.preprocessing_parameters))):
                # pp_step.undo_preprocessing(target_image.preprocessed_image, params, True)
                target_mask = pp_step.undo_preprocessing(target_mask, params)

            # Segmentation refinement (optional)
            if self.segmentation_refiner is not None:
                target_mask = self.segmentation_refiner.refine(target_mask, target_image)

            # Save segmentation
            target_segmentation_path = os.path.basename(target_image.image_path)[:-10] + "-mask.Gauss.png"
            self.save_segmentation(TargetSegmentation(target_segmentation_path, target_mask))



