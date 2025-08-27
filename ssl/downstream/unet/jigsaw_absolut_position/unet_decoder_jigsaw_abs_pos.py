import os

import torch
import torch.nn as nn

from ssl.downstream.unet.jigsaw_absolut_position.mobile_net_v2_attention_unet_decoder import \
    MobileNetV2AttentionUNetDecoder
from ssl.downstream.unet.jigsaw_absolut_position.mobile_net_v2_encoder_wrapper import MobileNetV2EncoderWrapper
from ssl.pretext.jigsaw_absolute_position.JigsawAbsPosNetwork import JigsawAbsPosNetwork


class UnetDecoderJigsawAbsPos(nn.Module):
    def __init__(self, pretext_model_path=None, num_classes=1, input_size=(256, 256), pretext_classes=9):
        super().__init__()
        # Load the pre-trained Jigsaw Absolute Position model
        pretext_model = JigsawAbsPosNetwork(num_positions=pretext_classes)
        # Load weights onto CPU to save GPU memory if the model is large
        if pretext_model_path and os.path.isfile(pretext_model_path):
            state = torch.load(pretext_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            pretext_model.load_state_dict(state, strict=False)
        else:
            # No pretext model weights provided or file does not exist; using random initialization
            pass
        # Create the encoder wrapper using the features of the loaded model
        self.encoder = MobileNetV2EncoderWrapper(pretext_model.features)

        # Determine encoder channel dimensions dynamically
        print("Determining encoder channel dimensions dynamically...")
        self.encoder.eval() # for dummy pass
        with torch.no_grad():
            # Create a dummy image with the expected input size
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            encoder_features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in encoder_features]
        print(f"--> Detected channels: {encoder_channels}")
        self.encoder.train() # Reset encoder back to training mode

        # Create the appropriate decoder with the dynamic dimensions
        self.decoder = MobileNetV2AttentionUNetDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        encoder_features = self.encoder(x)
        segmentation_mask = self.decoder(encoder_features)
        return segmentation_mask