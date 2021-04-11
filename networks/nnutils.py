import torch
import torch.nn as nn

def conv_unit(in_ch, out_ch, kernel_size, stride = 1, padding = 0, activation = 'relu', batch_norm = True):
    seq_list = []
    seq_list.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))

    if batch_norm:
        seq_list.append(nn.BatchNorm2d(num_features = out_ch))
    
    if activation == 'relu':
        seq_list.append(nn.ReLU())
    elif activation == 'sigmoid':
        seq_list.append(nn.Sigmoid())
    
    return nn.Sequential(*seq_list)