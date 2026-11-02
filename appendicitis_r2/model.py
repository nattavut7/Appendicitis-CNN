"""R2 custom architecture matching described building-block types.

Important: this is a transparent reference RE-IMPLEMENTATION. Per-layer
parameter allocations were not fully specified in the R1 paper, and thus
an exact reproduction of its original trained network cannot be claimed.
"""
from __future__ import annotations
import torch
from torch import nn


def round_channels(channels, width, divisor=2):
    return max(divisor, int(round(channels * width / divisor)) * divisor)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, se=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel, stride=stride, padding=kernel//2,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.PReLU(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
            SEBlock(out_ch) if se else nn.Identity())

    def forward(self, x):
        return self.net(x)


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, width_multiplier=0.75,
                 grouped_conv_groups=2, spatial_kernel=5, dropout=0.2,
                 classifier_hidden=2008):
        super().__init__()
        if spatial_kernel % 2 != 1 or spatial_kernel < 3:
            raise ValueError('spatial_kernel must be an odd integer >=3')
        c1, c2, c3, c4 = [round_channels(x, width_multiplier, grouped_conv_groups)
                          for x in (32, 128, 256, 320)]
        if any(c % grouped_conv_groups for c in (c1, c2)):
            raise ValueError('Channels must be divisible by grouped_conv_groups')
        self.conv1 = DepthwiseSeparable(in_channels, c1, 3, stride=2, se=True)
        self.conv2 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1,
                                             groups=grouped_conv_groups, bias=False),
                                   nn.BatchNorm2d(c2), nn.PReLU(c2))
        self.conv3 = nn.Sequential(nn.Conv2d(c2, c3, 1, bias=False),
                                   nn.BatchNorm2d(c3), nn.PReLU(c3), SEBlock(c3))
        self.conv4 = DepthwiseSeparable(c3, c4, spatial_kernel, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c4, classifier_hidden),
                                        nn.PReLU(), nn.Dropout(dropout),
                                        nn.Linear(classifier_hidden, num_classes))

    def forward_features(self, x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))

    def forward(self, x):
        return self.classifier(self.avgpool(self.forward_features(x)))

    def gradcam_target(self):
        return self.conv4


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_model(config):
    name = config['model'].get('architecture', 'r2_cnn')
    if name == 'r2_cnn':
        return LightweightCNN(in_channels=int(config['data']['channels']),
                              width_multiplier=float(config['model']['width_multiplier']),
                              grouped_conv_groups=int(config['model']['grouped_conv_groups']),
                              spatial_kernel=int(config['model']['spatial_kernel']),
                              dropout=float(config['model']['dropout']),
                              classifier_hidden=int(config['model']['classifier_hidden']))
    from torchvision import models
    constructors = {'mobilenet_v2': models.mobilenet_v2,
                    'efficientnet_b0': models.efficientnet_b0,
                    'efficientnet_b2': models.efficientnet_b2,
                    'inception_v3': models.inception_v3, 'vgg19': models.vgg19}
    if name not in constructors:
        raise ValueError(f'Unknown architecture: {name}')
    # No pretrained weights: avoids undocumented pretraining advantage.
    if name == 'inception_v3':
        network = constructors[name](weights=None, aux_logits=False)
        network.fc = nn.Linear(network.fc.in_features, 2)
    elif name.startswith('efficientnet') or name == 'mobilenet_v2':
        network = constructors[name](weights=None)
        last = network.classifier[-1]
        network.classifier[-1] = nn.Linear(last.in_features, 2)
    else:
        network = constructors[name](weights=None)
        network.classifier[-1] = nn.Linear(network.classifier[-1].in_features, 2)
    if config['data']['channels'] != 3:
        raise ValueError('Torchvision baselines require 3 input channels')
    return network
