from tsai.models.layers import *

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        super().__init__()
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        nf = 64
        self.nf = nf
        kss=[7, 5, 3]
        self.encoder = nn.Sequential(
            ResBlock(len(hparams.data.channels), nf, kss=kss),
            ResBlock(nf, nf * 2, kss=kss),
            ResBlock(nf * 2, nf * 2, kss=kss),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.fc = nn.Linear(nf * 2, len(hparams.data.labels))

    def forward(self, x):
        x = self.encoder(x)
        x = self.squeeze(self.gap(x))
        return self.fc(x)
