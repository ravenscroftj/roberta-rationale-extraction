"""
Multi-layer text CNN layer

This is shamelessly nicked from yala's implementation with a few small changes:
https://github.com/yala/text_nn/blob/476d1336f5be7178bc13b70a569a1a0b964b8244/rationale_net/models/cnn.py
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

from typing import List


class CNN(nn.Module):

    def __init__(self, embedding_dim: int,
                 num_layers: int,
                 max_pool_over_time: bool = False,
                 filters: List[int] = [3, 4, 5],
                 filter_num: int=100, cuda=True):

        super(CNN, self).__init__()

        if cuda and torch.has_cuda:
            self.cuda = True

        self.layers = []
        for layer in range(num_layers):
            convs = []
            for filt in filters:
                in_channels = embedding_dim
                kernel_size = filt
                new_conv = nn.Conv1d(
                    in_channels=in_channels, out_channels=filter_num, kernel_size=kernel_size)
                self.add_module('layer_'+str(layer) +
                                '_conv_'+str(filt), new_conv)
                convs.append(new_conv)

            self.layers.append(convs)

        self.max_pool = max_pool_over_time

    def _conv(self, x):
        layer_activ = x
        for layer in self.layers:
            next_activ = []
            for conv in layer:
                left_pad = conv.kernel_size[0] - 1
                pad_tensor_size = [d for d in layer_activ.size()]
                pad_tensor_size[2] = left_pad
                left_pad_tensor =autograd.Variable( torch.zeros( pad_tensor_size ) )
                if self.cuda:
                    left_pad_tensor = left_pad_tensor.cuda()
                padded_activ = torch.cat((left_pad_tensor, layer_activ), dim=2)
                next_activ.append(conv(padded_activ))

            # concat across channels
            layer_activ = F.relu(torch.cat(next_activ, 1))

        return layer_activ

    def _pool(self, relu):
        pool = F.max_pool1d(relu, relu.size(2)).squeeze(-1)
        return pool

    def forward(self, x):
        activ = self._conv(x)
        if self.max_pool:
            activ = self._pool(activ)
        return activ
