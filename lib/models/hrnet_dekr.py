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

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import HighResolutionModule
from .conv_block import BasicBlock, Bottleneck, AdaptBlock

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'ADAPTIVE': AdaptBlock
}


class PoseHigherResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.SPEC
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        self.num_limbs = cfg.DATASET.NUM_LIMBS
        self.keypoint_groups = cfg.DATASET.KEYPOINT_GROUPS
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i+1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i+2), stage)

        # build head net
        # 计算骨干网络的输出的通道数
        inp_channels = int(sum(self.stages_spec.NUM_CHANNELS[-1]))
        # 热图的配置信息
        config_heatmap = self.spec.HEAD_HEATMAP
        # 偏移量的配置信息
        config_offset = self.spec.HEAD_OFFSET
        # 关节点数量
        self.num_joints = cfg.DATASET.NUM_JOINTS
        # 偏移量的通道数量
        self.num_offset = self.num_joints * 2
        # 关节点加中心点的数量（热图数量）
        self.num_joints_with_center = self.num_joints+1
        # 每个分支输入的通道数
        self.offset_prekpt = config_offset['NUM_CHANNELS_PERKPT']

        # 检测头一共需要多少通道数
        offset_channels = self.num_joints*self.offset_prekpt
        # 调整输入通道的层
        self.transition_heatmap = self._make_transition_for_head(
                    inp_channels, config_heatmap['NUM_CHANNELS'])
        self.transition_offset = self._make_transition_for_head(
                    inp_channels, offset_channels)

        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(config_offset)
        self.offset_limbs_feature_layers, self.offset_limbs_final_layer = \
            self._make_separete_limbs_regression_head(config_offset)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)
        
        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'],
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_separete_limbs_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for i in range(self.num_limbs):
            # 用来生成肢体中心点偏移量的自适应卷积
            # 输入和输出是：肢体中关键点的个数乘以每个关键点特征图的层数
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT']*len(self.keypoint_groups[i]),
                layer_config['NUM_CHANNELS_PERKPT']*len(self.keypoint_groups[i]),
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT']*len(self.keypoint_groups[i]),
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        """

        Args:
            block:每个 layer 里面使用的 block，可以是 BasicBlock，Bottleneck
            inplanes:
            planes:输出的通道数
            blocks:一个整数，表示该层 layer 有多少个 block
            stride: 第一个 block 的卷积层的 stride，默认为 1。注意，只有在每个 layer 的第一个 block 的第一个卷积层使用该参数。
            dilation:

        Returns:

        """
        downsample = None
        # 判断 stride 是否为 1，输入通道和输出通道是否相等。如果这两个条件都不成立，那么表明需要建立一个 1 X 1 的卷积层，来改变通道数和改变图片大小。
        # 如果 stride不等于1，那么在建立bottleneck时,stride也是1，那么残差得出的 w和h就变化了，那么也需要把通过downsample的w和h处理一下，变为一致
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))
        # * 指的是python中传参解压功能
        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                     multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels
    def _get_index_by_val(self, val):
        dim1, dim2 = 99, 99
        for i, ks in enumerate(self.keypoint_groups):
            for j, value in enumerate(ks):
                if value == val:
                    dim1, dim2 = i, j
                    break
        return dim1, dim2



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i+1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i+2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat([y_list[0], \
            F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')], 1)

        heatmap = self.head_heatmap[1](
            self.head_heatmap[0](self.transition_heatmap(x)))

        final_offset = []
        offset_feature = self.transition_offset(x)
        # 用于存放肢体的输出特征图
        limbs_out_feature = []
        # 用于存放肢体的输出的偏移量
        final_limbs_offset = []
        # 循环肢体个数
        for i in range(self.num_limbs):
            limbs_in_feature = []
            # 遍历每个肢体中包含的关键点索引
            for idx in self.keypoint_groups[i]:
                # 根据关键点索引获取肢体中心点的输入特征图
                limbs_in_feature.append(offset_feature[:, idx*self.offset_prekpt:(idx+1)*self.offset_prekpt])
            # 合并处理
            limbs_in_feature = torch.cat(limbs_in_feature, dim=1)
            # 求得输出特征图
            limbs_out_feature.append(self.offset_limbs_feature_layers[i](limbs_in_feature))
            # 求得每个肢体中心点最终的偏移量
            final_limbs_offset.append(self.offset_limbs_final_layer[i](limbs_out_feature[i]))
        # 合并最终偏移量
        limbs_offset = torch.cat(final_limbs_offset, dim=1)

        # 将肢体中心关键点处理的特征图作为求关键点偏移量的输入特征，但是由于顺序不一样，所以要调整一下数据。
        # for i in range(self.num_limbs):
        #     for idx, keypoint_idx in enumerate(self.keypoint_groups[i]):
        #          offset_feature[:, keypoint_idx*self.offset_prekpt:(keypoint_idx+1)*self.offset_prekpt] = \
        #              limbs_out_feature[i][:, idx*self.offset_prekpt:(idx+1)*self.offset_prekpt]

        for j in range(self.num_joints):
            # 通过值获取索引
            dim1, dim2 = self._get_index_by_val(j)
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        limbs_out_feature[dim1][:, dim2*self.offset_prekpt:(dim2+1)*self.offset_prekpt])))

        offset = torch.cat(final_offset, dim=1)
        return heatmap, offset, limbs_offset

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 给可形变卷积中的卷积核和偏移量单独做参数初始化。
        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)
        # 模型中需要保存下来的参数包括两种:一种是反向传播需要被optimizer更新的，称之为 parameter
        # 一种是反向传播不需要被optimizer更新，称之为 buffer
        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            # 使用预训练权重加载模型。
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
