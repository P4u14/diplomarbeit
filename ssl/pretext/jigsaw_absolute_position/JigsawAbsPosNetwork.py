import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class JigsawAbsPosNetwork(nn.Module):
    def __init__(self, num_positions=9):
        super(JigsawAbsPosNetwork, self).__init__()
        # Initialize MobileNetV2 with a width multiplier of 0.75
        base_model = mobilenet_v2(weights=None, width_mult=0.75)

        # Reduce the output channels of the last convolutional layer to 512
        last_conv_in_channels = base_model.features[-1][0].in_channels
        base_model.features[-1][0] = nn.Conv2d(last_conv_in_channels, 512, kernel_size=1, stride=1, bias=False)

        # Adjust the subsequent batch norm layer
        base_model.features[-1][1] = nn.BatchNorm2d(512)

        self.features = base_model.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # The feature size from MobileNetV2's feature extractor output is now 512 (instead of 1280)
        feature_size = 512

        # A single classifier head that will be applied to the combined features
        # of the central patch and each peripheral patch.
        # The input size is feature_size * 2 because we concatenate two feature vectors.
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_positions)
        )

    def forward(self, x):
        # Input x shape: (B, T, C, H, W), where T is he number of tiles (e.g. 9 or 25)
        B, T, C, H, W = x.shape

        # Reshape to (B*T, C, H, W) to process all patches in a batch
        x = x.view(B * T, C, H, W)

        # Extract features for all patches
        features = self.features(x)

        # Apply global average pooling
        pooled_features = self.pool(features).view(B * T, -1)

        # Reshape back to (B, T, feature_size)
        pooled_features = pooled_features.view(B, T, -1)

        # The middle patch is at the center of the grid
        middle_idx = T // 2
        central_features = pooled_features[:, middle_idx, :].unsqueeze(1) # Shape: (B, 1, F)

        outputs = []
        for i in range(T):
            if i == middle_idx:
                continue

            peripheral_features = pooled_features[:, i, :] # Shape: (B, F)

            # Concatenate central features (broadcasted) with peripheral features
            combined_features = torch.cat([central_features.squeeze(1), peripheral_features], dim=1) # Shape: (B, 2*F)

            # Apply the classifier
            output = self.classifier(combined_features) # Shape: (B, num_positions)
            outputs.append(output)

        # Stack the outputs for the 8 peripheral positions
        # The output shape will be (B, 8, num_positions), e.g., (batch_size, 8, 9)
        return torch.stack(outputs, dim=1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
