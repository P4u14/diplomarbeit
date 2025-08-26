import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl.downstream.unet.attention_gate import AttentionGate


class UNetDecoderJigsawPerm(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetDecoderJigsawPerm, self).__init__()

        # Up-convolutional layers
        self.up_conv4 = nn.ConvTranspose2d(256, 384, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(384, 256, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(256, 96, kernel_size=2, stride=2)

        # Attention gates
        self.ag4 = AttentionGate(F_g=384, F_l=384, F_int=192)
        self.ag3 = AttentionGate(F_g=384, F_l=384, F_int=192)
        self.ag2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.ag1 = AttentionGate(F_g=96, F_l=96, F_int=48)

        # Convolutional blocks for decoder
        self.conv_block4 = nn.Conv2d(384 + 384, 384, kernel_size=3, padding=1)
        self.conv_block3 = nn.Conv2d(384 + 384, 384, kernel_size=3, padding=1)
        self.conv_block2 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        self.conv_block1 = nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1)

        # Final classifier
        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, encoder_features):
        e1_out, e2_out, e3_out, e4_out, e5_out = encoder_features

        # Decoder path
        d4 = self.up_conv4(e5_out)
        d4 = F.interpolate(d4, size=e4_out.shape[2:], mode='bilinear', align_corners=False)
        s4 = self.ag4(g=d4, x=e4_out)
        d4 = torch.cat((s4, d4), dim=1)
        d4 = self.conv_block4(d4)

        d3 = self.up_conv3(d4)
        d3 = F.interpolate(d3, size=e3_out.shape[2:], mode='bilinear', align_corners=False)
        s3 = self.ag3(g=d3, x=e3_out)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.conv_block3(d3)

        d2 = self.up_conv2(d3)
        d2 = F.interpolate(d2, size=e2_out.shape[2:], mode='bilinear', align_corners=False)
        s2 = self.ag2(g=d2, x=e2_out)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.conv_block2(d2)

        d1 = self.up_conv1(d2)
        d1 = F.interpolate(d1, size=e1_out.shape[2:], mode='bilinear', align_corners=False)
        s1 = self.ag1(g=d1, x=e1_out)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.conv_block1(d1)

        return self.final_conv(d1)
