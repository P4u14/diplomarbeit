from torch import nn


class AlexNetEncoderWrapper(nn.Module):
    """
    This wrapper takes an AlexNet model and breaks it down into 5 stages to provide the outputs for the skip connections
    in a U-Net architecture.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Break down the encoder into stages
        self.stage1 = nn.Sequential(*list(self.encoder.children())[0:4])
        self.stage2 = nn.Sequential(*list(self.encoder.children())[4:8])
        self.stage3 = nn.Sequential(*list(self.encoder.children())[8:10])
        self.stage4 = nn.Sequential(*list(self.encoder.children())[10:12])
        self.stage5 = nn.Sequential(*list(self.encoder.children())[12:15])

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return [s1, s2, s3, s4, s5]