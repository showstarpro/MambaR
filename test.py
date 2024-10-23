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
import misc as misc

# log about
import mlflow
import timm.optim.optim_factory as optim_factory

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

model_without_ddp = model
param_groups = optim_factory.add_weight_decay(model_without_ddp, 0.05)
optimizer = torch.optim.AdamW(param_groups, lr=0.0003, betas=(0.9, 0.95))
loss_scaler = NativeScaler()

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加参数
parser.add_argument('--output_dir', type=str, default="./test", help='Output directory')

# 解析参数
args = parser.parse_args()

misc.save_model(
    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    loss_scaler=loss_scaler, epoch=0)