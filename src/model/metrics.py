import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Callable, Sequence, Union, List, Dict
import math




class SphericalDistance(nn.Module):
    """Computes the mean spherical distance using the spherical law of cosines.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output. Assumes the first 8 elements are pairs of (longitude, latitude).
            target (torch.Tensor) (N, 12): The ground truth. Same structure as output.
        Returns:
            mean_distance (torch.Tensor) (0): The mean spherical distance.
        """
        

        long_lat_output = output[:, :8].reshape(-1, 2) # Reshape to (N*4, 2)
        long_lat_target = target[:, :8].reshape(-1, 2) # Reshape to (N*4, 2)

        # Convert degrees to radians
        long_lat_output = torch.deg2rad(long_lat_output)
        long_lat_target = torch.deg2rad(long_lat_target)

        # Calculate differences
        delta_long = long_lat_output[:, 1] - long_lat_target[:, 1]
        phi1, phi2 = long_lat_output[:, 0], long_lat_target[:, 0]

        # Spherical law of cosines
        delta_sigma = torch.acos(torch.sin(phi1) * torch.sin(phi2) + torch.cos(phi1) * torch.cos(phi2) * torch.cos(delta_long))

        # Return the mean spherical distance
        mean_distance = delta_sigma.mean()
        return mean_distance

# class SphericalDistance(nn.Module):
#     """Computes the mean spherical distance using the spherical law of cosines.
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, output, target):
#         """
#         Args:
#             output (torch.Tensor) (N, 12): The model output. Assumes the first 8 elements are pairs of (longitude, latitude).
#             target (torch.Tensor) (N, 12): The ground truth. Same structure as output.
#         Returns:
#             mean_distance (torch.Tensor) (0): The mean spherical distance.
#         """
#         n_pair = 4
#         output_copy = output.clone()
#         target_copy = target.clone()
#         output_copy = output_copy[:, :8]
#         target_copy = target_copy[:, :8]
#         # output_copy[:, :8:2] = output_copy[:, :8:2] * 180 - 90
#         # output_copy[:, 1:8:2] = output_copy[:, 1:8:2] * 360 - 180
#         # target_copy[:, :8:2] = target_copy[:, :8:2] * 180 - 90
#         # target_copy[:, 1:8:2] = target_copy[:, 1:8:2] * 360 - 180

#         # Extract longitude and latitude pairs from output and target
#         # Assuming the first 8 elements are pairs of (longitude, latitude)
#         long_lat_output = output_copy[:, :8].reshape(-1, 2) # Reshape to (N*4, 2)
#         long_lat_target = target_copy[:, :8].reshape(-1, 2) # Reshape to (N*4, 2)

#         # Convert degrees to radians
#         long_lat_output = torch.deg2rad(long_lat_output)
#         long_lat_target = torch.deg2rad(long_lat_target)

#         # Calculate differences
#         delta_long = long_lat_output[:, 1] - long_lat_target[:, 1]
#         phi1, phi2 = long_lat_output[:, 0], long_lat_target[:, 0]

#         # Spherical law of cosines
#         delta_sigma = torch.acos(torch.sin(phi1) * torch.sin(phi2) + torch.cos(phi1) * torch.cos(phi2) * torch.cos(delta_long))

#         # Return the mean spherical distance
#         mean_distance = delta_sigma.mean()
#         return mean_distance


class MeanSquareError(nn.Module):
    """Computes the mean squared error.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output.
            target (torch.Tensor) (N, 12): The target values.
        Returns:
            loss (torch.Tensor) (0): The MSE loss.
        """
        loss = (output - target) ** 2
        return loss.mean()