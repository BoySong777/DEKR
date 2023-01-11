# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch

from .CrowdPoseDataset import CrowdPoseDataset

logger = logging.getLogger(__name__)


class CrowdPoseKeypoints(CrowdPoseDataset):
    def __init__(self, cfg, dataset, heatmap_generator,
                 joints_offset_generator=None, limbs_offset_generator=None, transforms=None):
        super().__init__(cfg, dataset)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints + 1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.joints_offset_generator = joints_offset_generator
        self.limbs_offset_generator = limbs_offset_generator
        self.transforms = transforms

        self.ids = [
            img_id
            for img_id in self.ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]
        # 2022年12月31日17:12:01 增加肢体相关参数
        self.num_limbs = cfg.DATASET.NUM_LIMBS
        self.keypoint_groups = cfg.DATASET.KEYPOINT_GROUPS

    def __getitem__(self, idx):
        img, anno, image_info = super().__getitem__(idx)

        mask = self.get_mask(anno, image_info)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]
        joints, area = self.get_joints(anno)
        limbs, limbs_area = self._get_limbs_point(joints)

        joints = np.array([np.vstack((obj, limbs[i])) for i, obj in enumerate(joints)])
        area = np.hstack((area, limbs_area))

        if self.transforms:
            img, mask_list, joints_list, area = self.transforms(
                img, [mask], [joints], area
            )
        # limbs = joints_list[0][:, -5:, :]
        # limbs_area = area[:, -5:]
        joints_list_without_limbs = [joints_list[0][:, :-5, :]]
        # area = area[:, :1]

        heatmap, ignored = self.heatmap_generator(
            joints_list_without_limbs[0], self.sigma, self.center_sigma, self.bg_weight)
        mask = mask_list[0] * ignored

        # offset, offset_weight = self.offset_generator(joints_list[0], area)
        offset, offset_weight = self.joints_offset_generator(joints_list[0], area)
        limbs_offset, limbs_weight = self.limbs_offset_generator(joints_list[0], area)

        return img, heatmap, mask, offset, offset_weight, limbs_offset, limbs_weight

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints_with_center, 3))

        for i, obj in enumerate(anno):
            # 将keypoint从一维数组转换为2维数组
            joints[i, :self.num_joints, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            area[i, 0] = self.cal_area_2_torch(
                torch.tensor(joints[i:i + 1, :, :]))

            joints_sum = np.sum(joints[i, :-1, :2], axis=0)
            num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
            if num_vis_joints <= 0:
                # 如果关节点数量小于等于0的话，那么人的中心关键点为0，0。
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 1

        return joints, area

    def get_mask(self, anno, img_info):
        m = np.zeros((img_info['height'], img_info['width']))

        return m < 0.5

    def _get_limbs_point(self, joints):
        """
        用来一个获取输入图片中的每个人的肢体中心点
        Args:
            joints: （人数，关节点数，3）

        Returns:（人数，5，3）

        """
        # 获取人数
        num_people = len(joints)

        # 初始化每个肢体的面积
        area = np.zeros((num_people, self.num_limbs))
        # 初始化肢体中心点的矩阵
        limbs = np.zeros((num_people, self.num_limbs, 3))
        # 以人为单位，遍历每个人
        for i, obj in enumerate(joints):
            # 遍历关节点分组，以获得每个肢体的关节点。
            for j, index in enumerate(self.keypoint_groups):
                # 求得每个肢体的面积（面积的平方）
                area[i, j] = self.cal_area_2_torch(
                    torch.tensor(joints[i:i + 1, index, :]))
                # 求得每个肢体中包含的关键点的总和
                limb_sum = np.sum(joints[i, index, :2], axis=0)
                # 求得被标记的关键点的个数
                num_vis_joints = len(np.nonzero(joints[i, index, 2])[0])
                # 如果没有可见关键点，那么肢体中心点也为零。
                if num_vis_joints <= 0:

                    limbs[i, j, :2] = 0
                    limbs[i, j, 2] = 0
                else:
                    limbs[i, j, :2] = limb_sum / num_vis_joints
                    limbs[i, j, 2] = 1

        return limbs, area
