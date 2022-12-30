# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import tools._init_paths
import torch

from config import cfg
from config import update_config
from dataset import make_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.TRAIN.IMAGES_PER_GPU = 1
    # cfg.DATASET.DATASET = "crowd_pose_kpt"
    cfg.freeze()
    # 创建数据加载器，其中distributed参数数在设置是否使用多块GPU进行分布式训练。
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=False
    )
    for i, (image, heatmap, mask, offset, offset_w) in enumerate(train_loader):
        print(type(image))
        print("*" * 50)
        image = image.cpu().numpy()
        print("image:\n")
        print("*"*50)
        print("heatmap:\n" )
        heatmap = heatmap.cpu().numpy()
        print(heatmap)
        print("*" * 50)
        print("mask:\n")
        print(mask)
        print("*" * 50)
        print("offset:\n")
        print(offset)
        print("*" * 50)
        print("offset_w:\n" )
        print(offset_w)
        print("*" * 50)

if __name__ == '__main__':
    main()