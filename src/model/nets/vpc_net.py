import torch
import torch.nn as nn
import torchvision.models as models
from src.model.nets.base_net import BaseNet

# torch.autograd.set_detect_anomaly(True)


# class VPClusterNet(BaseNet):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         # Load a pre-trained ResNet (excluding its final fully connected layer)
#         self.resnet = models.resnet34(pretrained=True)  # You can change to resnet34, resnet50, etc.

#         # Replace the first convolutional layer if in_channels is not 3 (RGB)
#         if in_channels != 3:
#             self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # Remove the average pooling and fully connected layers
#         self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))

#         # Fully connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512*512, 256),  # 512 is the output feature size of ResNet18/34, for ResNet50/101/152 use 2048
#             nn.ReLU(inplace=True),
#             nn.Linear(256, out_channels),
#             nn.Sigmoid()  # Sigmoid to get outputs in the range [0, 1]
#         )

#     def forward(self, x):
#         # Feature extraction
#         features = self.resnet(x)

#         # Fully connected layers for classification
#         output = self.fc_layers(features)
#         return output

class VPClusterNet(BaseNet):
    def __init__(self, in_channels):
        super().__init__()
        # Load a pre-trained ResNet (excluding its final fully connected layer)
        self.resnet = models.resnet50(pretrained=True)
        
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the average pooling and fully connected layers
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        # Determine flattened size for feature vector
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 512, 1024)
            dummy_output = self.resnet(dummy_input)
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]

        # Branch for eight-dimensional vector
        self.fc_branch_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8),
            # nn.Sigmoid()
        )

        # # Branch for four-dimensional vector
        # self.fc_branch_2 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(flattened_size, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 4), 
        #     # nn.Softmax(dim=1)
        # )
        self.fc_branch_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4), 
            # nn.Softmax(dim=1)
        )

        # Freeze the weights of the ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Initialize weights of the fc layers
        for branch in [self.fc_branch_1, self.fc_branch_2]:
            for module in branch.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight.data)
                    nn.init.zeros_(module.bias.data)

    def forward(self, x):
        # Feature extraction
        features = self.resnet(x)

        # Separate branches
        pitch_yaw_output = self.fc_branch_1(features)
        real_numbers_output = self.fc_branch_2(features)

        # Clip the pitch and yaw
        # clipped_pitch = torch.clamp(pitch_yaw_output[:, 0:8:2], 0, 180)
        # pclipped_yaw = torch.clamp(pitch_yaw_output[:, 1:8:2], 0, 360)
        # normalize pitch and yaw
        # pitch_yaw = torch.cat((clipped_pitch, pclipped_yaw), dim=1)

        # Concatenate outputs from both branches
        combined_output = torch.cat((pitch_yaw_output, real_numbers_output), dim=1)
        return combined_output