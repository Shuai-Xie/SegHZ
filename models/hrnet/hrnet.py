# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from inplace_abn.abn import InPlaceABNSync
import functools

# 多 GPU 使用
# BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')  # identity = none
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1
ALIGN_CORNERS = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """与 resnet 同样的 BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """与 resnet 同样的 Bottleneck
    inplanes 降维到 planes, 完成 3x3 conv 加速计算; 再升维到原始维度 inplanes
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 降维
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 3x3 conv 计算 (减少 planes 加速)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)  # relu 共用，因为没有可学习的参数
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self,
                 num_branches,
                 block, num_blocks,
                 num_inchannels, num_channels,
                 fuse_method,  # sum / cat
                 multi_scale_output=True):
        """
        1.构建 branch 并行 多 scale 特征提取
        2.在 module 末端将 多 scale 特征通过 upsample/downsample 方式，并用 sum 进行 fuse
            注意：这里的 sum fuse 是值从 多个 branch(j) 到 branch_i 的聚合结果;
                 整个 module 的输出结果依然是 并行的 num_branch 个结果
        :param num_branches: stage 并行高度
        :param block: BASIC/BOTTLENECK
        :param num_blocks: 指定每个 block 重复次数
        :param num_inchannels: 由 NUM_CHANNELS 和 block.expansion 相乘得到
        :param num_channels:
        :param fuse_method: sum / cat
        :param multi_scale_output:
        """
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, block, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        """
        一个分支的 Sequential 结构
        :param branch_index: 第几个 branch
        :param block: 类型
        :param num_blocks: 重复次数, cfg 每个 branch 设置的次数都 = 4
        :param num_channels: channel
        :param stride:
        :return:
        """
        # 判断是否是 stage 连接处
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))

        # stage 内部后几个 block
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],  # inplanes
                                num_channels[branch_index]))  # planes

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """
        并行分支的 ModuleList 结构
        :param num_branches: 分支数
        :param block: BASIC/BOTTLENECK
        :param num_blocks: 每个分支 block 重复次数
        :param num_channels: 每个分支 channel
        :return:
        """
        branches = []

        for i in range(num_branches):
            branches.append(  # add one branch, 内部 features, stride=1
                self._make_one_branch(i, block, num_blocks, num_channels, stride=1))

        return nn.ModuleList(branches)  # 使用 ModuleList 得到并行分支结果

    def _make_fuse_layers(self):
        """
        混合 branch 输出结果，得到 fusion 特征
        :return:
        fuse ModuleList(): 每个 branch 都会输出一组 生成不同大小 output 的 Sequential
            [
                branch1 ModuleList(),  1/4  -> [1/4, 1/8, 1/16]
                branch2 ModuleList(),  1/8  -> [1/4, 1/8, 1/16]
                branch3 ModuleList(),  1/16 -> [1/4, 1/8, 1/16]
            ]
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:  # ↗, 深 -> 浅, 通道转换，上采样 (forward 完成)
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i],  # 通道转换
                                  1, 1, 0, bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:  # → 同层
                    fuse_layer.append(None)
                else:  # ↘, 浅 -> 深, 下采样
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:  # 下采样次数
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j],
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_inchannels[j], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        # stage1
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # stage 2/3/4
        # 并行得到每个 branch 结果，仍存入 x list
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # fuse, stage 内融合方式默认就采用了 sum
        x_fuse = []
        for i in range(len(self.fuse_layers)):  # 每个 branch
            # 其他 branch(j) 结果 fuse 到 branch_i 的结果
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):  # 对应转到的每个 branch
                if i == j:
                    y = y + x[j]
                elif j > i:  # 上采样
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
                                          size=[x[i].shape[-2], x[i].shape[-1]],  # H,W
                                          mode='bilinear', align_corners=ALIGN_CORNERS)
                else:  # 下采样
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        """
        :param config: 以 yaml 存储的 模型配置文件，确定模型的结构
        :param kwargs: 其他关键字参数
        """
        extra = config.MODEL.EXTRA  # 对应 yaml 文件 MODEL.EXTRA 下参数，和 config['MODEL']['EXTRA'] 效果一样
        super(HighResolutionNet, self).__init__()

        # stem net
        # 两层 3x3 conv，stride=2，得到 1/4 大小的 feature map
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        # 开始 HRModule 阶段
        # 每个 stage 不仅保留之前所有 size 的特征，还增加一个新的下采样 size 特征
        # stage1: [1/4]
        # stage2: [1/4, 1/8]
        # stage3: [1/4, 1/8, 1/16]
        # stage4: [1/4, 1/8, 1/16, 1/32]

        # stage1
        self.stage1_cfg = extra['STAGE1']  # 取 stage 1 cfg 文件
        block = blocks_dict[self.stage1_cfg['BLOCK']]  # BOTTLENECK, block 类型
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]  # 64, 通道数
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]  # 4, stage 内 block 重复次数
        # make layer, layer1 同 resnet
        self.layer1 = self._make_layer(
            block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels  # BOTTLENECK 4*64, 输出通道数

        # stage2
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # [48,96]
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # BASIC
        num_channels = [  # [48,96] 并行 block 对应的 输出通道数
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # transition1
        # 将 stage1 output 转成 stage2 需要的两种输入尺寸和通道数 [1/4] -> [1/4, 1/8]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        # make stage, 构建每个 stage 并行交叉的网络结构
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)  # [48,96]

        # stage3
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # channel sum 48,96,192,384 = 720
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        # concat 后缓冲
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.cls_head = nn.Conv2d(in_channels=last_inp_channels, out_channels=config.DATASET.NUM_CLASSES,
                                  kernel_size=extra.FINAL_CONV_KERNEL,
                                  stride=1,  # 这里用 1x1 conv 代替 fc
                                  padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)  # FINAL_CONV_KERNEL=1
        self.init_weights()

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
        :param num_channels_pre_layer: pre_stage output channels list
        :param num_channels_cur_layer: cur_stage output channels list
            cur 总比 pre 多一个 output_channel 对应增加的 1/2 下采样
                    stage2      stage3          stage4
            pre:    [256]       [48,96]         [48,96,192]
            cur:    [48,96]     [48,96,192]     [48,96,192,384]

            每个 stage channels 数量也对应了 stage2/3/4 使用 BASIC block; expansion=1
        :return:
            transition_layers:
                1.完成 pre_layer 到 cur_layer input channels 数量对应
                2.完成 feature map 尺寸对应
        """
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)
        # num_branches_cur - num_branches_pre = 1

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(  # 降采样 pre 到 cur num_channels
                        nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                                      BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                                      nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                # cur 中最后一个 channel，需要下采样
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):  # 对应 3x3,stride=2 卷积使用次数，下采样次数
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),  # stride=2
                                                  BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                                                  nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        # ModuleList 内部 module 可以索引使用，从而实现 transition 需要的并行运算
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """
        :param block: BasicBlock / Bottleneck
        :param inplanes: 输入通道数
        :param planes: 中间通道数
        :param blocks: layer 内 block 重复次数
        :param stride: 步长 >1 说明 layer 连接处有下采样，需要 downsample
        :return:
        """
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # stride=1 and inplanes == planes * block.expansion; 为 layer 内部 block
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """
        创建 num_modules 个 HighResolutionModule 结构，每个 module 末尾完成 hrnet 特有的特征融合模块
        :param layer_config:
        :param num_inchannels: 由 NUM_CHANNELS 和 block.expansion 相乘得到
        :param multi_scale_output: 都是 True
        :return:
            HighResolutionModule 结构
                        stage2  stage3  stage4
        num_modules:     1       4       3
        """
        # eg. stage2
        num_modules = layer_config['NUM_MODULES']  # 1, HighResolutionModule 重复次数
        num_branches = layer_config['NUM_BRANCHES']  # 2, 并行分支数，高度
        num_blocks = layer_config['NUM_BLOCKS']  # [4,4]，每个分支 block 重复次数
        num_channels = layer_config['NUM_CHANNELS']  # [48,96]，每个分支 channels
        block = blocks_dict[layer_config['BLOCK']]  # BASIC
        fuse_method = layer_config['FUSE_METHOD']  # SUM，multi scale 特征融合方式

        modules = []
        for i in range(num_modules):  # 添加 num_modules 个 HighResolutionModule
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,  # 高度
                                     block,  # BASIC/BOTTLENECK
                                     num_blocks,  # 宽度
                                     num_inchannels,  # block feature 宽度
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()  # cls method

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        _, _, H, W = x.shape

        # stem net
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)  # 1/4

        # stage1
        x = self.layer1(x)  # 1/4

        # stage2
        # prepare input
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):  # 2，并行分支数
            if self.transition1[i] is not None:  # 非 None, 需要 channel or size 转换
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)  # 同 channel & size, 直接添加
        # forward
        y_list = self.stage2(x_list)

        # stage3
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:  # 非
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))  # 对 stage2 最小 size 特征下采样，并完成通道转换
            else:
                x_list.append(y_list[i])  # 同 channel & size, 直接添加之前 stage 结果
        y_list = self.stage3(x_list)

        # stage4
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))  # 下采样
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)  # 1/4
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        # HRNetV2, concat all features
        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.final_conv(x)  # 1/4
        x = self.cls_head(x)

        return x

    def init_weights(self):
        print('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, InPlaceABNSync) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrain(self, pretrained):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and 'last_layer' not in k}  # 不加载最后的 head 参数
            # for k, v in pretrained_dict.items():
            #     print('=> loading {} | {}'.format(k, v.size()))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print('load done!')
        else:
            print('No such file {}'.format(pretrained))


def HRNet(cfg_path, num_classes, use_pretrain=True, **kwargs):
    from models.hrnet.config import update_config

    cfg = update_config(cfg_path, num_classes)
    model = HighResolutionNet(cfg, **kwargs)
    if use_pretrain:
        model.load_pretrain(cfg.MODEL.PRETRAINED)

    return model


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/home/xs/codes/SegHZ')

    model = HRNet(f'models/hrnet/cfg/seg_hrnet_w48.yaml', 7, use_pretrain=True)
    x = torch.randn(1, 3, 512, 512)
    res = model(x)
    print(res.shape)
