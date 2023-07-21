"""
Author: Liu
Time: 2023.06.14
"""
import torch
from torch import nn


class GaitModelNN(nn.Module):
    r"""Class for gait neural network modules.

        Should subclass this class.

        Note:
            An ``__init__()``
        """
    def __init__(self):
        super(GaitModelNN, self).__init__()
        self.conv_1st = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0, dilation=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0, dilation=1),
        )
        self.flatten_1st = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 30, out_features=16, bias=True),
        )
        self.outputs_1st = nn.Sequential(
            nn.Linear(in_features=16, out_features=3, bias=True),
        )

    def forward(self, inputs):
        temp = self.conv_1st(inputs)
        temp = self.flatten_1st(temp)
        outputs = self.outputs_1st(temp)
        return outputs

