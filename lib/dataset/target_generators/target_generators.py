# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints_with_center = num_joints+1

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0):
        # joints的shape应该是 人数*关节点数*2（猜测）
        assert self.num_joints_with_center == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints_with_center

        hms = np.zeros((self.num_joints_with_center, self.output_res, self.output_res),
                       dtype=np.float32)
        # 这个是在弄关键点的权重，在关键点周围权重为1
        # 初始值全部初始化为2
        ignored_hms = 2*np.ones((self.num_joints_with_center, self.output_res, self.output_res),
                                    dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for p in joints:
            for idx, pt in enumerate(p):
                if idx < 17:
                    sigma = sgm
                else:
                    sigma = ct_sgm
                # 判断关键点的可见性
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    # 判断关键点是否为正常值
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue
                    #求出进行高斯平滑的范围，以关节点为中心点
                    ul = int(np.floor(x - 3 * sigma - 1)
                             ), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 2)
                             ), int(np.ceil(y + 3 * sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx -
                                     cc] = self.get_heat_val(sigma, sx, sy, x, y)

                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][idx, aa:bb, cc:dd] = 1.

        # 将剩余的权重的改为指定权重。
        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints+1
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius

    def __call__(self, joints, area):
        """

        Args:
            joints: 关节点位置
            area: 人的一个面积的平方

        Returns: 人的每个关节点的偏移量，和一个相对应位置的一个权重，权重是这个人面积的倒数。

        """
        assert joints.shape[1] == self.num_joints_with_center, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue
            # 从头开始遍历，遍历到最后一个（不包含最后一个）为止
            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                # 如果这个位置已经有值了，且这个值对应的人的面积小于当前的人的面积，这这个值就不变了，
                                # 这说明，如果一个位置被两个人占用，那么就让面积小的人占用
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            area_map[pos_y, pos_x] = area[person_id]

        return offset_map, weight_map

class JointsOffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius, limbs_groups=None):
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints+1
        self.num_limbs = 5
        self.num_joints_with_center_and_limbs = num_joints + 1 + self.num_limbs
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius
        self.limbs_groups = limbs_groups

    def __call__(self, joints, area):
        """

        Args:
            joints: 关节点位置
            area: 人的一个面积的平方

        Returns: 人的每个关节点的偏移量，和一个相对应位置的一个权重，权重是这个人面积的倒数。

        """
        assert joints.shape[1] == self.num_joints_with_center_and_limbs, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, p in enumerate(joints):
            for limb_id, limb_index in enumerate(self.limbs_groups):
                ct_x = int(p[self.num_joints_with_center + limb_id, 0])
                ct_y = int(p[self.num_joints_with_center + limb_id, 1])
                ct_v = int(p[self.num_joints_with_center + limb_id, 2])
                if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                        or ct_x >= self.output_w or ct_y >= self.output_h:
                    continue
                # 从头开始遍历，遍历到最后一个（不包含最后一个）为止
                for idx, pt in enumerate(p[limb_index]):
                    # idx 是关键点的索引
                    idx = limb_index[idx]
                    if pt[2] > 0:
                        x, y = pt[0], pt[1]
                        if x < 0 or y < 0 or \
                                x >= self.output_w or y >= self.output_h:
                            continue

                        start_x = max(int(ct_x - self.radius), 0)
                        start_y = max(int(ct_y - self.radius), 0)
                        end_x = min(int(ct_x + self.radius), self.output_w)
                        end_y = min(int(ct_y + self.radius), self.output_h)

                        for pos_x in range(start_x, end_x):
                            for pos_y in range(start_y, end_y):
                                offset_x = pos_x - x
                                offset_y = pos_y - y
                                if offset_map[idx*2, pos_y, pos_x] != 0 \
                                        or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                    # 如果这个位置已经有值了，且这个值对应的人的面积小于当前的人的面积，这这个值就不变了，
                                    # 这说明，如果一个位置被两个人占用，那么就让面积小的人占用
                                    if area_map[pos_y, pos_x] < area[person_id][0]:
                                        continue
                                offset_map[idx*2, pos_y, pos_x] = offset_x
                                offset_map[idx*2+1, pos_y, pos_x] = offset_y
                                weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id][0])
                                weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id][0])
                                area_map[pos_y, pos_x] = area[person_id][0]

        return offset_map, weight_map

class LimbsOffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints+1
        self.num_limbs = 5
        self.num_joints_with_center_and_limbs = num_joints+1+self.num_limbs
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius

    def __call__(self, joints, area):
        """

        Args:
            joints: 关节点位置
            area: 人的肢体的一个面积的平方

        Returns: 人的每个关节点的偏移量，和一个相对应位置的一个权重，权重是这个人面积的倒数。

        """
        assert joints.shape[1] == self.num_joints_with_center_and_limbs, \
            'the number of joints should be 18, 17 keypoints + 1 center joint + 5 limb center joints.'

        offset_map = np.zeros((self.num_limbs * 2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_limbs*2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, p in enumerate(joints):
            ct_x = int(p[-6, 0])
            ct_y = int(p[-6, 1])
            ct_v = int(p[-6, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue
            # 从倒数第5个开始遍历，遍历到最后一个（包含最后一个）为止，因为最后5个存的是肢体的中心点
            for idx, pt in enumerate(p[-5:]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id][idx+1]:
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id][idx+1])
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id][idx+1])
                            area_map[pos_y, pos_x] = area[person_id][idx]

        return offset_map, weight_map
