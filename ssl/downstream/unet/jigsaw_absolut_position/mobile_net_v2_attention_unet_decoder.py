import torch
from torch import nn
import torch.nn.functional as F

from ssl.downstream.unet.attention_gate import AttentionGate


class MobileNetV2AttentionUNetDecoder(nn.Module):
    """
    This decoder is specifically designed to process feature maps from the
    MobileNetV2EncoderWrapper. It is dynamically configured based
    on the channel dimensions of the encoder.
    """
    def __init__(self, encoder_channels, num_classes=9):
        super().__init__()

        # Unpack the dynamically determined channel dimensions
        s1_c, s2_c, s3_c, s4_c, s5_c = encoder_channels

        # Stage 4
        self.up_conv4 = nn.ConvTranspose2d(s5_c, s4_c, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=s4_c, F_l=s4_c, F_int=s4_c // 2)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(s4_c * 2, s4_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(s4_c, s4_c, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        # Stage 3
        self.up_conv3 = nn.ConvTranspose2d(s4_c, s3_c, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=s3_c, F_l=s3_c, F_int=s3_c // 2)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(s3_c * 2, s3_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(s3_c, s3_c, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        # Stage 2
        self.up_conv2 = nn.ConvTranspose2d(s3_c, s2_c, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=s2_c, F_l=s2_c, F_int=s2_c // 2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(s2_c * 2, s2_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(s2_c, s2_c, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        # Stage 1
        self.up_conv1 = nn.ConvTranspose2d(s2_c, s1_c, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=s1_c, F_l=s1_c, F_int=s1_c // 2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(s1_c * 2, s1_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(s1_c, s1_c, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        # Final convolution to generate the mask
        self.final_conv = nn.Conv2d(s1_c, num_classes, kernel_size=1)

    def forward(self, encoder_features):
        e1_out, e2_out, e3_out, e4_out, e5_out = encoder_features

        # Decoder path
        d4 = self.up_conv4(e5_out)
        # Optional: Interpolate if sizes don't match perfectly due to padding
        if d4.shape[2:] != e4_out.shape[2:]:
            d4 = F.interpolate(d4, size=e4_out.shape[2:], mode='bilinear', align_corners=False)
        s4 = self.ag4(g=d4, x=e4_out)
        d4 = torch.cat((s4, d4), dim=1)
        d4 = self.conv_block4(d4)

        d3 = self.up_conv3(d4)
        if d3.shape[2:] != e3_out.shape[2:]:
            d3 = F.interpolate(d3, size=e3_out.shape[2:], mode='bilinear', align_corners=False)
        s3 = self.ag3(g=d3, x=e3_out)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.conv_block3(d3)

        d2 = self.up_conv2(d3)
        if d2.shape[2:] != e2_out.shape[2:]:
            d2 = F.interpolate(d2, size=e2_out.shape[2:], mode='bilinear', align_corners=False)
        s2 = self.ag2(g=d2, x=e2_out)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.conv_block2(d2)

        d1 = self.up_conv1(d2)
        if d1.shape[2:] != e1_out.shape[2:]:
            d1 = F.interpolate(d1, size=e1_out.shape[2:], mode='bilinear', align_corners=False)
        s1 = self.ag1(g=d1, x=e1_out)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.conv_block1(d1)

        return self.final_conv(d1)