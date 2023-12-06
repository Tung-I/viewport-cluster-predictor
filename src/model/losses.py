import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Callable, Sequence, Union, List, Dict
import math


# class WeightedMSELoss(nn.Module):
#     """Weighted Mean Squared Error Loss.
#     """
#     def __init__(self):
#         super().__init__()
#         # Define weights for each entry in the 12-dimensional vector
#         self.weights = torch.tensor([0.3/4, 0.4/4, 0.3/4, 0.4/4, 0.3/4, 0.4/4, 0.3/4, 0.4/4, 0.3/4, 0.3/4, 0.3/4, 0.3/4])

#     def forward(self, output, target):
#         """
#         Args:
#             output (torch.Tensor) (N, 12): The model output.
#             target (torch.Tensor) (N, 12): The target values.
#         Returns:
#             loss (torch.Tensor) (0): The weighted MSE loss.
#         """
#         if self.weights.device != output.device:
#             self.weights = self.weights.to(output.device)

#         output_copy = output.clone()
#         target_copy = target.clone()
#         output_copy[:, :8:2] = output_copy[:, :8:2] / 90
#         output_copy[:, 1:8:2] = output_copy[:, 1:8:2] / 180
#         target_copy[:, :8:2] = target_copy[:, :8:2] / 90
#         target_copy[:, 1:8:2] = target_copy[:, 1:8:2] / 180

#         # Calculate the weighted MSE loss
#         loss = self.weights * (output_copy - target_copy) ** 2
#         return loss.mean()

class SphericalDistanceLoss(nn.Module):
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
        # The last four elements of the target are the weights of the four pairs
        # For instance, the target[n, 8:12] = [0.3, 0.2, 0.3, 0.2]
        # then the weights of the first eight loss terms are [0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2]
        probs = target[:, 8:]

        # print(f'output: {output[0]}')
        # print(f'target: {target[0]}')
    
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
        # Reshape delta_sigma to (N, 4)
        delta_sigma = delta_sigma.reshape(-1, 4)
        # Apply the weights
        delta_sigma = delta_sigma * probs
        delta_sigma = delta_sigma.sum(dim=1)
        # Return the mean spherical distance
        mean_distance = delta_sigma.mean()

        return mean_distance

class ProbabilityLoss(nn.Module):
    """Computes Cross-Entropy Loss for probability predictions.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output. The last 4 elements are class probabilities.
            target (torch.Tensor) (N, 12): The ground truth. The last 4 elements are the true class labels.
        Returns:
            loss (torch.Tensor) (0): The Cross-Entropy Loss for probability predictions.
        """
        # Assuming the last four elements are probabilities and the target class is the one with the highest probability
        probs_output = output[:, 8:]
        target_classes = target[:, 8:].max(dim=1)[1]  # Get the index of the max probability

        loss = self.loss_fn(probs_output, target_classes)
        return loss



class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error Loss.
    """
    def __init__(self):
        super().__init__()
        # Define weights for each entry in the 12-dimensional vector
        self.weight_py = 1.0
        self.weight_prob = 0

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output.
            target (torch.Tensor) (N, 12): The target values.
        Returns:
            loss (torch.Tensor) (0): The weighted MSE loss.
        """
        # scale the output[0, 2, 4, 6] by 90 and output[1, 3, 5, 7] by 180
        output_copy = output.clone()
        target_copy = target.clone()
        # output_copy[:, :8:2] = output_copy[:, :8:2] / 90
        # output_copy[:, 1:8:2] = output_copy[:, 1:8:2] / 180
        # target_copy[:, :8:2] = target_copy[:, :8:2] / 90
        # target_copy[:, 1:8:2] = target_copy[:, 1:8:2] / 180

        # print(output)
        # print(target)
        # raise Exception
        # Calculate the weighted MSE loss
        # Extract the (pitch-yaw) pairs from the 12-dimensional vector

        # The loss of the first 4 pairs is weighted by their corresponding ground truth probabilities

        prob_weights = target_copy[:, 8:]
        # duplicate the prob_weights for each pair
        prob_weights = torch.repeat_interleave(prob_weights, 2, dim=1)

        prob_weighted_loss = prob_weights * (output_copy[:, :8] - target_copy[:, :8]) ** 2
        # summate over all pairs
        prob_weighted_loss = prob_weighted_loss.sum(dim=1)

    

        # loss = self.weight_py * prob_weighted_loss + self.weight_prob * prob_loss

        # loss = self.weights * (output - target) ** 2
        # return loss.mean()
        return prob_weighted_loss.mean()
