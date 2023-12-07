import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Callable, Sequence, Union, List, Dict
import math


class MyMSELoss(nn.Module):
    """Mean Squared Error Loss.
    """
    def __init__(self):
        super().__init__()
        # Define weights for each entry in the 12-dimensional vector

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output.
            target (torch.Tensor) (N, 12): The target values.
        Returns:
            loss (torch.Tensor) (0): The weighted MSE loss.
        """

        # print for debugging
        # print(f'output: {output[0]}')
        # print(f'target_scaled: {target_scaled[0]}')
        # print(f'output: {output}')
        # print(f'target: {target}')
        loss = (output - target) ** 2
        # print(f'loss: {loss}')
        # print(f'loss: {loss.mean()}')
        return loss.mean()

class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error Loss.
    """
    def __init__(self):
        super().__init__()
        # Define weights for each entry in the 12-dimensional vector
        self.weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output.
            target (torch.Tensor) (N, 12): The target values.
        Returns:
            loss (torch.Tensor) (0): The weighted MSE loss.
        """
        if self.weights.device != output.device:
            self.weights = self.weights.to(output.device)

        # Calculate the weighted MSE loss
        loss = self.weights * (output - target) ** 2
        return loss.mean()


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



# class WeightedMSELoss(nn.Module):
#     """Weighted Mean Squared Error Loss.
#     """
#     def __init__(self):
#         super().__init__()
#         # Define weights for each entry in the 12-dimensional vector

#     def forward(self, output, target):
#         """
#         Args:
#             output (torch.Tensor) (N, 12): The model output.
#             target (torch.Tensor) (N, 12): The target values.
#         Returns:
#             loss (torch.Tensor) (0): The weighted MSE loss.
#         """

#         target_scaled = target.clone()
#         target_scaled[:, :8:2] = target_scaled[:, :8:2] / 90
#         target_scaled[:, 1:8:2] = target_scaled[:, 1:8:2] / 180  

#         # print(f'output: {output[0]}')
#         # print(f'target: {target[0]}')
#         # print(f'target_scaled: {target_scaled[0]}')

#         prob_weighted_loss = (output[:, 0:8:2]- target_scaled[:, 0:8:2]) ** 2 
#         # summate over all pairs
#         # prob_weighted_loss = prob_weighted_loss.sum(dim=1)

    

#         # loss = self.weight_py * prob_weighted_loss + self.weight_prob * prob_loss

#         # loss = self.weights * (output - target) ** 2
#         # return loss.mean()
#         return prob_weighted_loss.mean()
