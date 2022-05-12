# Discriminator code
# All of Networks in this experiment share same Discriminator but different weights
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tvmodels

class Discriminator(nn.Module):
    def __init__(self,use_bias=False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)

        )

    def forward(self, input):
        output = self.layers(input)
        return output

class Discriminator_Mod(nn.Module):
    def __init__(self, leaky_relu_negative_slope=0.2, use_bias=False):
        super().__init__()

        self.negative_slope = leaky_relu_negative_slope
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)

        )

    def forward(self, input):
        output = self.layers(input)
        return output