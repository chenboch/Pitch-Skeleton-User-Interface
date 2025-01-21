#!/usr/bin/python
# -*- coding:utf8 -*-

# model
from .build import build_model, get_model_hyperparameter

# DSTA
from .DSTA.dsta_std_resnet50 import DSTA_STD_ResNet50

from .DSTA.dsta_std_resnet50_gau import DSTA_STD_ResNet50_GAU


from .DSTA.dsta_std_vit import DSTA_STD_Vit

from .DSTA.dsta_std_vitm import DSTA_STD_VitM

from .DSTA.dsta_std_hrnet import DSTA_STD_HrNet

# HRNet
from .backbones.hrnet import HRNet

# SimpleBaseline
from .backbones.simplebaseline import SimpleBaseline
