import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),  # 降低 feature 维度
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)  # 并行结构

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            feat = f(x)
            out.append(F.interpolate(feat, x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)  # 与原始 feature 又 concat 一起


if __name__ == '__main__':
    x = torch.randn(1, 960, 128, 128)

    in_dim = 960
    bins = [1, 2, 3, 6]
    model = PPM(in_dim, reduction_dim=in_dim // 8, bins=bins)  # reduction_dim 决定是否 lightweight
    model.eval()

    res = model(x)
    print(res.shape)
