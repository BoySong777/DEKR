# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data
# 测试用
from .COCODataset import CocoDataset as coco
from .COCODataset import CocoRescoreDataset as rescore_coco
# 训练用
from .COCOKeypoints import CocoKeypoints as coco_kpt
# 测试用
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseDataset import CrowdPoseRescoreDataset as rescore_crowdpose
# 训练用
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import OffsetGenerator
from .target_generators import JointsOffsetGenerator
from .target_generators import LimbsOffsetGenerator


def build_dataset(cfg, is_train):
    """
    该函数用来完成dataset实例化。
    Args:
        cfg:
        is_train:

    Returns:

    """
    assert is_train is True, 'Please only use build_dataset for training.'

    # 对图片进行一些变换操作，如翻转，仿射变换等。
    transforms = build_transforms(cfg, is_train)

    #
    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS
    )
    offset_generator = OffsetGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS
    )
    joints_offset_generator = JointsOffsetGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS, limbs_groups=cfg.DATASET.KEYPOINT_GROUPS
    )
    limbs_offset_generator = LimbsOffsetGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS
    )

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.TRAIN,
        heatmap_generator,
        joints_offset_generator,
        limbs_offset_generator,
        transforms
    )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        # 每块GPU一次处理多少张图片
        # 也就是说每一次计算，计算多少张图片，这个和batch还不一样。
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        # 将数据集打乱
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    # 每一个batch可以处理多少张图片，计算方式是每块GPU上处理的照片数量 乘以 GPU的块数。
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS, # 多线程处理数据
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg, cfg.DATASET.TEST
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
