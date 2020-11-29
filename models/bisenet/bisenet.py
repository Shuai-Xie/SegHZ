import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base import ConvBlock, DeconvBlock, SEBlock
from models.bisenet.build_contextpath import build_contextpath
from models.bisenet.build_contextpath_dcn import build_contextpath_dcn
from modules.attention.PAM import PAM



class Spatial_path(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.sp = nn.Sequential(
            ConvBlock(3, out_dim // 4, kernel_size=3, stride=2, padding=1),
            ConvBlock(out_dim // 4, out_dim // 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(out_dim // 2, out_dim, kernel_size=3, stride=2, padding=1)  # 1/8
        )

    def forward(self, x):
        return self.sp(x)


# FFM
class FeatureFusionModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # compress feature
        self.compress_conv = ConvBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        # channel attention
        self.se = SEBlock(out_dim, reduction=8)  # 256/8=32

    def forward(self, x1, x2, size):
        x1 = F.interpolate(x1, size=size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=size, mode='bilinear', align_corners=True)
        # fusion feature, 不同感受野 feature
        x = torch.cat((x1, x2), dim=1)  # todo: 只压缩高维特征, 低维保留
        x = self.compress_conv(x)
        out = x + self.se(x)  # 残差连接
        return out


class BiSeNet(nn.Module):
    def __init__(self, num_classes, context_path, in_planes=64, use_dcn=False):
        super().__init__()

        sp_dim, ffm_out_dim = 256, 256
        fuse_dim = sp_dim + ffm_out_dim
        # if ffm_out_dim == num_classes:
        #     deconv_dims = [num_classes] * 3
        # else:
        #     deconv_dims = [ffm_out_dim, ffm_out_dim // 2, ffm_out_dim // 4]

        if context_path == 'resnet18':
            arm_dims = [in_planes * 4, in_planes * 8]
            ffm_dim = sum(arm_dims)
        elif context_path == 'resnet50' or context_path == 'resnet101':
            arm_dims = [in_planes * 4 * 4, in_planes * 8 * 4]  # expansion=4
            ffm_dim = sum(arm_dims)
        else:
            raise NotImplementedError

        self.saptial_path = Spatial_path(sp_dim)
        if use_dcn:
            print('use dcn')
            self.context_path = build_contextpath_dcn(context_path, in_planes, output_stride=16, pretrained=True)
        else:
            self.context_path = build_contextpath(context_path, in_planes, output_stride=16, pretrained=True)

        # middle features
        # channel attention
        self.arm1 = SEBlock(arm_dims[0])
        self.arm2 = SEBlock(arm_dims[1])

        # middle supervision
        self.mid1 = nn.Conv2d(arm_dims[0], num_classes, kernel_size=1)
        self.mid2 = nn.Conv2d(arm_dims[1], num_classes, kernel_size=1)

        # features fusion + feature compress + channel attention
        self.ffm = FeatureFusionModule(ffm_dim, ffm_out_dim)  # cx1 + cx2

        self.seg_head = nn.Sequential(
            ConvBlock(fuse_dim, fuse_dim, kernel_size=1, padding=0),  # concat feature -> fuse
            PAM(fuse_dim, reduction=8),  # space attention, out8 feature 尺寸变大 pam 耗时
            DeconvBlock(fuse_dim, fuse_dim, kernel_size=4, stride=2, padding=1),  # 1/4
            nn.Conv2d(fuse_dim, num_classes, kernel_size=1)  # 最后再 conv
        )
        # x8 -> decoder, IoU 提升 0.6
        # 成倍上采样 k - s = 2 pad
        # 逐步减少 dim https://www.programcreek.com/python/example/107696/torch.nn.ConvTranspose2d
        # self.deconv = nn.Sequential(
        # DeconvBlock(deconv_dims[0], deconv_dims[1], kernel_size=4, stride=2, padding=1),  # 1/4
        # DeconvBlock(deconv_dims[1], deconv_dims[1], kernel_size=4, stride=2, padding=1),
        # DeconvBlock(deconv_dims[1], deconv_dims[2], kernel_size=4, stride=2, padding=1),
        # )
        # self.last_conv = nn.Conv2d(deconv_dims[-1], num_classes, kernel_size=1)  # 最后再 conv

    def forward(self, x):
        sp = self.saptial_path(x)
        cx1, cx2 = self.context_path(x)
        cx1, cx2 = self.arm1(cx1), self.arm2(cx2)  # gap 已经在 arm 中做了，没必要再乘 tail
        # cx1, cx2 = self.pam1(cx1), self.pam2(cx2)

        cx = self.ffm(cx1, cx2, sp.shape[2:])  # 1/8

        # 尽可能保存 1/8 特征的作用
        fuse = torch.cat((sp, cx), dim=1)
        res = self.seg_head(fuse)

        # 单模型可用 self.training 判断状态
        if self.training:  # 使用 nn.Module 自带属性判断 training/eval 状态
            res1, res2 = self.mid1(cx1), self.mid2(cx2)  # 1/16, 1/16
            return [res1, res2, res]  # 1/16, 1/16, 1/4
        else:
            return res
