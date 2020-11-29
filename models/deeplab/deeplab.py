import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))  # 添加项目目录，python2 不支持中文

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplab.backbone import build_backbone
from models.deeplab.aspp import build_aspp
from models.deeplab.decoder import build_decoder


class DeepLab(nn.Module):

    def __init__(self, backbone='resnet50', output_stride=16, num_classes=11,
                 sync_bn=False, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8  # 1/8

        if sync_bn:  # 多 GPU
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        # eval is freeze
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():  # conv, bn
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


@torch.no_grad()
def cmp_infer_time(test_num=20):
    import time
    archs = ['mobilenet', 'resnet50', 'resnet101']

    x = torch.rand(1, 3, 512, 512)
    x = x.cuda()

    for arch in archs:
        model = DeepLab(arch, num_classes=37)
        model.cuda()
        model.eval()

        torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成, CPU 等待 cuda 所有运算执行完才退出
        t1 = time.time()
        for _ in range(test_num):
            model(x)
        t2 = time.time()
        torch.cuda.synchronize()

        t = (t2 - t1) / test_num
        fps = 1 / t

        # print(f'{arch} - {inp} \t time: {t} \t fps: {fps}')
        print('{} \t time: {} \t fps: {}'.format(arch, t, fps))


if __name__ == '__main__':
    cmp_infer_time()
