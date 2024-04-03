from __future__ import absolute_import

import torch.nn as nn
import torch
import matplotlib.pyplot as plt


__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):

    def __init__(self):
        super(_AlexNet, self).__init__()
        self.n_layers = 1
        self.hidden_size = 45*45
        self.num_directions = 1
        self.hidden = None
        self.init_hidden()

    def init_hidden(self, batch_size=8):
        # Initialize hidden state with zeros
        self.hidden = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size)
        return self.hidden

    def forward(self, x, search=False, reset_hidden=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if search:

            # reset hidden in case of new video
            if reset_hidden is not None:
                reset_indices = torch.nonzero(reset_hidden).view(-1)
                if len(reset_indices) > 0:
                    self.hidden[:, reset_indices, :] = 0

            x = x.view(8, 32, -1)
            output, self.hidden = self.rnn(x, self.hidden)
            self.hidden = self.hidden.detach()  # remove it from computation graph
            output = output.view(8, 32, 45, 45)
            return output

        output = x
        return output


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

        # 2025 is 45x45 which is the search image features
        self.rnn = nn.GRU(2025, self.hidden_size, self.n_layers, dropout=0.05, batch_first=True,
                          bidirectional=True if self.num_directions == 2 else False)


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))

        # 2025 is 45x45 which is the search image features
        self.rnn = nn.GRU(2025, self.hidden_size, self.n_layers, dropout=0.2, batch_first=True,
                          bidirectional=True if self.num_directions == 2 else False)


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))

        # 2025 is 45x45 which is the search image features
        self.rnn = nn.GRU(2025, self.hidden_size, self.n_layers, dropout=0.2, batch_first=True,
                          bidirectional=True if self.num_directions == 2 else False)
