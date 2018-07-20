# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class MultiLayerNeuralNetworks(nn.Module):
    """Multi-Layer neural networks."""
    def __init__(self, input_feature, hidden_size, num_classes, activation=nn.ReLU()):
        super(MultiLayerNeuralNetworks, self).__init__()
        self.input_layer = nn.Linear(input_feature, hidden_size)
        self.hidden = nn.Linear(hidden_size, num_classes)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        out = self.hidden(x)
        return out

