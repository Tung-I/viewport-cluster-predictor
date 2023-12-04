import torch
import torch.nn as nn
import torchvision.models as models

class VPClusterNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Load a pre-trained ResNet (excluding its final fully connected layer)
        self.resnet = models.resnet50(pretrained=True)  # You can change to resnet34, resnet50, etc.
        
        # Replace the first convolutional layer if in_channels is not 3 (RGB)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the average pooling and fully connected layers
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),  # 512 is the output feature size of ResNet18/34, for ResNet50/101/152 use 2048
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels),
            nn.Sigmoid()  # Sigmoid to get outputs in the range [0, 1]
        )

    def forward(self, x):
        # Feature extraction
        features = self.resnet(x)

        # Fully connected layers for classification
        output = self.fc_layers(features)
        return output
