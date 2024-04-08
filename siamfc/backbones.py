from __future__ import absolute_import

import torch.nn as nn
import torch
from siamfc_pytorch.siamfc.attention import SelfAttention
import torch.nn.functional as F
from siamfc_pytorch.siamfc.cbam import CBAM, ChannelAttention, SpatialAttention
from siamfc_pytorch.siamfc.double_attention import DoubleAttention

__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3', 'RecurrentModel', 'SpatialAttentionAlexNet',
           'CBAMAttentionAlexNet', 'DoubleAttentionAlexNet', 'SelfAttentionAlexNet',
           'RNNAttentionAlexNet', 'SeqFramesAttenAlexNet']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):
    def forward(self, x, search=False, reset_hidden=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


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


class RecurrentModel(AlexNetV1):
    def __init__(self):
        super(RecurrentModel, self).__init__()
        self.n_layers = 1
        self.hidden_size = 45 * 45
        self.num_directions = 1
        self.hidden = None
        self.init_hidden()

        # Custom Network Layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.GRU(400, self.hidden_size, self.n_layers, dropout=0.05, device="cuda",
                          batch_first=True, bidirectional=False)

    def init_hidden(self, batch_size=8):
        # Initialize hidden state with zeros
        self.hidden = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size).to('cuda')
        return self.hidden

    def forward(self, x, search=False, reset_hidden=None, prev_frame=None):
        x = super().forward(x)

        assert not torch.isnan(x).any(), "X input contains NaNs"

        if not search:
            return x

        if search:
            # reset hidden in case of new video
            if reset_hidden is not None:
                reset_indices = torch.nonzero(reset_hidden).view(-1)
                if len(reset_indices) > 0:
                    self.hidden[:, reset_indices, :] = 0

            # pass through rnn layer first :)
            x = x.view(8, 256, -1)
            x, self.hidden = self.rnn(x, self.hidden)
            self.hidden = self.hidden.detach()  # remove it from computation graph
            x = x.view(8, 256, 45, 45)
            return x


class SpatialAttentionAlexNet(AlexNetV1):
    def __init__(self):
        super(SpatialAttentionAlexNet, self).__init__()
        # Custom Network Layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # spatial attention
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.attn_conv = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1)

    def forward(self, x, search=False, reset_hidden=None):
        x = super().forward(x)

        assert not torch.isnan(x).any(), "X input contains NaNs"

        if not search:
            return x

        if search:
            maxp_x = self.max_pool(x)
            avgp_x = self.avg_pool(x)
            spat_attn = self.attn_conv(torch.cat([maxp_x, avgp_x], dim=1))
            new_x = torch.mul(spat_attn, x)
            new_x = F.silu(new_x)
            assert not torch.isnan(new_x).any(), "X input contains NaNs"
            return new_x


class CBAMAttentionAlexNet(AlexNetV1):
    def __init__(self):
        super(CBAMAttentionAlexNet, self).__init__()
        # Custom Network Layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.attn = CBAM(256, 1)

    def forward(self, x, search=False, reset_hidden=None):
        x = super().forward(x)

        assert not torch.isnan(x).any(), "X input contains NaNs"

        if not search:
            return x

        if search:
            x = self.attn(x)
            assert not torch.isnan(x).any(), "X input contains NaNs"
            return x


class DoubleAttentionAlexNet(AlexNetV1):
    def __init__(self):
        super(DoubleAttentionAlexNet, self).__init__()
        # Custom Network Layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.attn = DoubleAttention(256, 128, 128)

    def forward(self, x, search=False, reset_hidden=None):
        x = super().forward(x)

        assert not torch.isnan(x).any(), "X input contains NaNs"

        if not search:
            return x

        if search:
            x = self.attn(x)
            assert not torch.isnan(x).any(), "X input contains NaNs"
            return x


class SelfAttentionAlexNet(AlexNetV1):
    def __init__(self):
        super(SelfAttentionAlexNet, self).__init__()
        # Custom Network Layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.self_atten = SelfAttention(256)

    def forward(self, x, search=False, reset_hidden=None):
        x = super().forward(x)

        assert not torch.isnan(x).any(), "X input contains NaNs"

        if not search:
            return x

        if search:
            x = self.self_atten(x)
            assert not torch.isnan(x).any(), "X input contains NaNs"
            return x


class RNNAttentionAlexNet(RecurrentModel):
    def __init__(self):
        super(RNNAttentionAlexNet, self).__init__()
        # Custom Network Layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ).cuda()
        self.atten = SelfAttention(256).cuda()

        self.rnn = nn.GRU(400, self.hidden_size, self.n_layers, dropout=0.05,
                          batch_first=True, bidirectional=False).cuda()

    def forward(self, x, search=False, reset_hidden=None, prev_frame=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        assert not torch.isnan(x).any(), "X input contains NaNs"

        if not search:
            return x

        if search:
            # reset hidden in case of new video
            if reset_hidden is not None:
                reset_indices = torch.nonzero(reset_hidden).view(-1)
                if len(reset_indices) > 0:
                    self.hidden[:, reset_indices, :] = 0

            # pass through rnn layer first :)
            x = x.view(8, 256, -1)
            x, self.hidden = self.rnn(x, self.hidden)
            self.hidden = self.hidden.detach()  # remove it from computation graph
            x = x.view(8, 256, 45, 45)
            x = self.atten(x)
            return x


class SeqFramesAttenAlexNet(RecurrentModel):
    def __init__(self):
        super(SeqFramesAttenAlexNet, self).__init__()
        self.conv1 = self.conv1.cuda()
        self.conv2 = self.conv2.cuda()
        self.conv3 = self.conv3.cuda()
        self.conv4 = self.conv4.cuda()
        self.conv5 = self.conv5.cuda()

        self.atten = SelfAttention(256).cuda()

    def forward(self, x, search=False, reset_hidden=None, prev_frame=None):
        def _conv(val):
            val = self.conv1(val)
            val = self.conv2(val)
            val = self.conv3(val)
            val = self.conv4(val)
            val = self.conv5(val)
            return val

        if not search:
            x = _conv(x)
            assert not torch.isnan(x).any(), "X input contains NaNs"
            return x

        if search:
            x1, x2 = x[0], x[1]  # the current frame, the prev frame
            x1 = _conv(x1)
            x2 = _conv(x2)

            # reset hidden in case of new video
            if reset_hidden is not None:
                reset_indices = torch.nonzero(reset_hidden).view(-1)
                if len(reset_indices) > 0:
                    self.hidden[:, reset_indices, :] = 0

            # Reshape tensors to add a new dimension
            x = x1 - x2
            out = self.atten(x)
            return out
