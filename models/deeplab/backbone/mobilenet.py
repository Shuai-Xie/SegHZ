"""
MobileNetV2
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo
import constants


def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)  # PW 隐藏层 提升 features 数量
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            # v1 block, 不提升 in planes, DW + PW-linear
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            # v2 block, 提升 in planes, PW + DW + PW-linear
            # 纺锤形，先提升 channel(PW), 多 channel 计算 3x3 conv(DW), 再压缩 channel(PW)
            # 和 resnet 沙漏结构相反，但都是为了不改变 feature 数量，而改善 feature 品质
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),  # kernel,stride,pad,dilation
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


class MobileNetV2(nn.Module):

    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True, mc_dropout=False):
        """
        @param output_stride:
        @param BatchNorm:
        @param width_mult: 1. 是否让 网络变得更宽，作为 multi 因子 * input_channel
        @param pretrained:
        @param mc_dropout:
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual  # 纺锤形 反 residual 模块
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t(特征升维 expand_ratio), c(通道数), n(重复次数), s(stride)
            [1, 16, 1, 1],  # 2
            [6, 24, 2, 2],  # 4
            [6, 32, 3, 2],  # 8
            [6, 64, 4, 2],  # dilation=2
            [6, 96, 3, 1],
            [6, 160, 3, 2],  # dilation=4
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)

        # 所有 features in list
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]  # stride=2
        current_stride *= 2  # 2 before building interverted_residuals
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            # 根据 output_stride 选择 stride=2 由 conv 实现 还是 dilation
            if current_stride == output_stride:  # output_stride 下采样 rate
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):  # n: 重复次数
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                else:
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel

        if mc_dropout:  # last features, for MC train
            self.features.append(nn.Dropout2d(p=constants.MC_DROPOUT_RATE))

        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_model()

        # 直接截取到子 Sequential model
        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]
        self.dropout = nn.Dropout2d(p=constants.MC_DROPOUT_RATE)  # for MC test
        self.mc_dropout = mc_dropout

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)

        if self.mc_dropout:
            low_level_feat = self.dropout(low_level_feat)

        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    input = torch.rand(1, 3, 256, 256)
    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)

    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
