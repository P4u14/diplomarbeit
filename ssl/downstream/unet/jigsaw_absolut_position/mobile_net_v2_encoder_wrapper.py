from torch import nn


class MobileNetV2EncoderWrapper(nn.Module):
    """
    This wrapper takes the ‘features’ module of a MobileNetV2 model
    and breaks it down into 5 stages to provide the outputs for the skip connections
    in a U-Net architecture.
    """
    def __init__(self, original_features):
        super().__init__()

        # Division of feature extraction into stages based on downsampling points.
        # Channel numbers for width_mult=0.75:
        self.stage1 = nn.Sequential(*original_features[0:2]) # Output: 12 channels
        self.stage2 = nn.Sequential(*original_features[2:4]) # Output: 18 channels
        self.stage3 = nn.Sequential(*original_features[4:7]) # Output: 24 channels
        self.stage4 = nn.Sequential(*original_features[7:14]) # Output: 72 channels
        self.stage5 = nn.Sequential(*original_features[14:]) # Output: 512 channels

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return [s1, s2, s3, s4, s5]