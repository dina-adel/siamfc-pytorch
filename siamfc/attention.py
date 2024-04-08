import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, n_in, n_out, ks=1, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=ks, bias=bias).cuda()
        self.norm = None
        self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query, self.key, self.value = [self._conv(n_channels, c) for c in
                                            (n_channels // 8, n_channels // 8, n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self, n_in, n_out):
        return ConvLayer(n_in, n_out, ks=1, bias=False)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma.cuda() * torch.bmm(h, beta).cuda() + x
        return o.view(*size).contiguous()
