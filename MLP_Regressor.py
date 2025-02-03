import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import numpy as np

'''
Paper Notes: https://arxiv.org/abs/2106.11189

    Feed forward neural network, initialized to 5 layers
    Learning rate of 10^-2, chosen for cosine annealing scheduler
    Use AdamW optimizer instead of adam due to weight decay parameter
    Learning rate schedular: Cosine annealing with restarts
        Restarts have initial budget of 15 epochs, budget multiplier of 2

Regularization features implemented so far:

    Implicit: early stopping, batch normalization
    Weight decay: weight decay coefficient as part of AdamW optimizer
    Ensembles: snapshot ensemble, dropout rate
'''

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims = [256, 256, 256, 128, 64], learning_rate=1e-2, use_snapshots=False, cycle_length=15, cycle_mult=2, batch_normalization=False, weight_decay=0.01, dropout_rate=0.25, early_stopping_patience=30, activation_fn=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.use_snapshots = use_snapshots
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult 
        self.batch_normalization = batch_normalization
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.early_stopping_patience = early_stopping_patience

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            if (batch_normalization):
                layers.extend([
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    activation_fn,
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                ])
            else:
                layers.extend([
                    nn.Linear(dims[i], dims[i + 1]),
                    activation_fn,
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                ])
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self):
        return AdamW(self.parameters(), lr=self.learning_rate,  weight_decay=self.weight_decay)
    
    def get_scheduler(self, optimizer):
        return CosineAnnealingWarmRestarts(optimizer, T_0=self.cycle_length,
            T_mult=self.cycle_mult, eta_min=1e-6)

    # for given list of snapshots (state_dicts), load each state with parameters intitalized to this model
    def ensemble_load(self, snapshots, device):
        ensemble_models = []
        for state_dict in snapshots:
            model_copy = MLPRegressor(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                use_snapshots=self.use_snapshots,
                cycle_length=self.cycle_length,
                cycle_mult=self.cycle_mult,
                batch_normalization=self.batch_normalization,
                weight_decay=self.weight_decay,
                dropout_rate=self.dropout_rate,
                early_stopping_patience=self.early_stopping_patience,
            )

            model_copy.load_state_dict(state_dict)
            model_copy.to(device)
            ensemble_models.append(model_copy)

        return ensemble_models

    # use ensemble_load to build separate models, before returning predictions as a list
    def ensemble_predict(self, X, snapshots, device):
        ensemble_models = self.ensemble_load(snapshots, device)

        with torch.no_grad():
            predictions = []
            for model in ensemble_models:
                model.eval()
                predictions.append(model(X).squeeze())
            return predictions