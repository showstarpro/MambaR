# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma

# modified NativeScaler for accum_iter
# from timm.utils import NativeScaler
from utils import NativeScaler

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
from augment import new_data_aug_generator

from contextlib import suppress

import models_mamba
import models_star
# import models_deit

import utils

# log about
import mlflow


model = create_model(
    "star_base_patch16_224",
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate= 0.1,
    drop_block_rate=None,
    img_size=224
)

model = model.cuda()

inputs = torch.randn(2, 3, 224, 224)
inputs = inputs.cuda()


out = model(inputs, return_features=True)

print(out)
