import torch
import torch.nn as nn
import torchvision.models as models
from src.model.nets.base_net import BaseNet

# torch.autograd.set_detect_anomaly(True)

class VPClusterNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Load a pre-trained ResNet (excluding its final fully connected layer)
        self.resnet = models.resnet34(pretrained=True)  # You can change to resnet34, resnet50, etc.
        
        # Replace the first convolutional layer if in_channels is not 3 (RGB)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the average pooling and fully connected layers
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        # Calculate the size of the flattened features
        with torch.no_grad():
            # Dummy forward pass to determine output size
            dummy_input = torch.zeros(1, in_channels, 512, 1024)
            dummy_output = self.resnet(dummy_input)
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels),
            # nn.Sigmoid()
        )

        # Freeze the weights of the ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Initialize the fc layers
        for module in self.fc_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.zeros_(module.bias.data)

        self.softsign = nn.Softsign()

    def forward(self, x):
        # Feature extraction
        features = self.resnet(x)

        # Fully connected layers for classification
        output = self.fc_layers(features)

        # Adjusting pitch, yaw, and probability outputs
        pitch_yaw = output[:, :8]  # Scale to [-1, 1]

        # pitch_yaw = self.softsign(pitch_yaw)

        # clip pitch to [-90, 90] using torch.clamp[-90, 90]
        torch.clamp(pitch_yaw[:, 0:8:2], min=-90, max=90)
        # clip yaw to [-180, 180] using torch.clamp[-180, 180]
        torch.clamp(pitch_yaw[:, 1:8:2], min=-180, max=180)
        
        # scaled_yaw = pitch_yaw[:, 1:8:2] * 180
        # scaled_pitch = pitch_yaw[:, 0:8:2] * 90
        # print(scaled_pitch[0])
        # Reconstruct pitch_yaw tensor
        # pitch_yaw = torch.stack((scaled_pitch, scaled_yaw), dim=2).flatten(start_dim=1)

        # print(f'pitch_yaw: {pitch_yaw[0]}')
        # raise Exception

        probabilities = torch.sigmoid(output[:, 8:])  # Probabilities in [0, 1]


        return torch.cat((pitch_yaw, probabilities), dim=1)

    # def forward(self, x):
    #     # Feature extraction
    #     features = self.resnet(x)

    #     # Fully connected layers for classification
    #     output = self.fc_layers(features)

    #     return output
