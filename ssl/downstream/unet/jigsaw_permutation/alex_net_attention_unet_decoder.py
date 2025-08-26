from torch import nn

from ssl.downstream.unet.jigsaw_permutation.alex_net_encoder_wrapper import AlexNetEncoderWrapper
from ssl.downstream.unet.jigsaw_permutation.unet_decoder_jigsaw_perm import UNetDecoderJigsawPerm


class AlexNetAttentionUNetDecoder(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = AlexNetEncoderWrapper(encoder)
        self.decoder = UNetDecoderJigsawPerm(num_classes=num_classes)

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(encoder_features)
        return decoder_output