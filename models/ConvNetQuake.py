__all__ = ['ConvNetQuake']

import torch
from torch import nn
from collections import defaultdict
import lightning as L
import lightning.pytorch as pl
from importlib import import_module




class ConvNetQuake(L.LightningModule):
    def __init__(self, hparams:dict):
        super().__init__()
        self.hparams.update(hparams)
        self.encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LazyConv1d(out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.LazyBatchNorm1d(),
            ) for out_channels in hparams.model.encoder.out_channels],
            nn.Flatten(1,2),
        )
        self.fc = nn.Sequential(
            # nn.Linear(**hparams.model.fc_1.params),
            nn.LazyLinear(**hparams.model.fc_1.params),
            nn.ELU(),
            nn.LazyLinear(len(hparams.data.labels)),
        )


    def forward(self, X):
        # (Batch, Channel, Length)
        X = self.encoder(X)
        # (Batch, Channel)
        X = self.fc(X)
        # (Batch, Class)
        return X