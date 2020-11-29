import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        hidden_dim = in_dim // reduction
        self.se = nn.Sequential(
            # squeeze
            nn.AdaptiveAvgPool2d((1, 1)),  # GAP
            # exciation
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),  # in = out
            nn.BatchNorm2d(hidden_dim),  # 去掉 bn 就和 linear 方式一样了
            nn.ReLU(inplace=True),  # 已经有使网络稀疏的能力
            # 这里如果用 dropout，会使得学到的 channel att scalar 丢失?
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1),  # 保留这里的 bias
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class SELayer(nn.Module):
    # todo: 内部没有 BN 所以性能差?
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(inplace=True),  # 已经有使网络稀疏的能力, 没有bn
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # element-wise
