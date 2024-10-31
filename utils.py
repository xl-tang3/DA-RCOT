import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader


from torchvision import models
import numpy as np
import cv2
from PIL import Image
from torch import autograd
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt
def downsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = 1/scale_factor, mode='bicubic') # downsample
    return y

def upsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = scale_factor, mode='bicubic') # upsample
    return y

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
        
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def calculate_valid_crop_size(crop_size, upscale_factor):
    """Calculates size of largest crop, divisible by upscale factor."""
    if isinstance(crop_size, int):
        return crop_size - (crop_size % upscale_factor)
    else:
        crop_size_w, crop_size_h = crop_size
        valid_crop_size_w = crop_size_w - (crop_size_w % upscale_factor)
        valid_crop_size_h = crop_size_h - (crop_size_h % upscale_factor)
        return (valid_crop_size_w, valid_crop_size_h)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        x = x.squeeze()
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h = self._tensor_size(x[:, 1:])
        count_w = self._tensor_size(x[1:, :])
        w_tv = torch.pow((x[:, 1:]-x[:, :w_x-1]), 2).sum()
        h_tv = torch.pow((x[1:, :]-x[:h_x-1, :]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)