import sys

sys.path.insert(0, '/nfs/xs/tmp/DeepGlobe')

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchvision.models.resnet import model_urls
import math

from modules.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modules.dcn.dcn_v2 import DCN


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None, use_dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        # 3x3
        if use_dcn:
            # conv2.weight / conv2.bias 依然可用 pretrain
            self.conv2 = DCN(planes, planes, kernel_size=3, stride=stride,
                             dilation=dilation, padding=dilation, deformable_groups=1)
            self.conv2.bias.data.zero_()
            self.conv2.conv_offset_mask.weight.data.zero_()
            self.conv2.conv_offset_mask.bias.data.zero_()
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, inplanes,
                 output_stride=16, BatchNorm=nn.BatchNorm2d,
                 layers=[0, 0, 0, 0],
                 dcn_layers=[0, 0, 0, 0], dcn_interval=1,  # dcn layers 表示哪些 stage 会用 dcn conv
                 pretrained=True):

        super(ResNet, self).__init__()  # inplanes 默认 =64, 可调节小网络
        self.inplanes = inplanes
        self.BatchNorm = BatchNorm

        # before layers, out 1/4
        # layer3 2/1 for different output stride
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]  # 1/4, 1/8, 1/16, 1/16
        elif output_stride == 8:  # strides 少1个2; layer3,4, dilation x2
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1/4

        self.layer1 = self._make_layer(block, inplanes, layers[0], strides[0], dilations[0], dcn_layers[0], dcn_interval)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], strides[1], dilations[1], dcn_layers[1], dcn_interval)  # stage3
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], strides[2], dilations[2], dcn_layers[2], dcn_interval)  # stage4
        blocks = [1, 2]  # multi grids
        self.layer4 = self._make_MG_unit(block, inplanes * 8, blocks, strides[3], dilations[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model(layers)

    def _make_layer(self, block, planes, blocks, stride=1,
                    dilation=1,
                    dcn_layers=0, dcn_interval=1):
        """
        :param block: BasicBlock, Bottleneck
        :param planes: features num = planes * block.expansion
        :param blocks: block repeat times
        :param stride: 1st conv's stride of current layer
        :param dilation:
        :return:
        """
        # layer 连接处，首层残差连接 是否需要 downsample
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.BatchNorm(planes * block.expansion),
            )

        layers = []
        use_dcn = (dcn_layers >= blocks)
        # 首个 block
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, self.BatchNorm, use_dcn=use_dcn))
        # 内部 block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_dcn = ((i + dcn_layers) >= blocks) and (i % dcn_interval == 0)  # res101, dcn_interval=3
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=self.BatchNorm, use_dcn=use_dcn))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        """
        级联 dilation 模块，参考 deeplabv3+，维持 1/16, 但采集更大感受野 feature
        blocks: [1, 2] dilation=2 -> dilations: 2,4
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.BatchNorm(planes * block.expansion),
            )
        layers = []
        # use_dcn = (dcn_layers >= blocks)
        use_dcn = True
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=self.BatchNorm, use_dcn=use_dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=self.BatchNorm, use_dcn=use_dcn))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 1/4
        x = self.layer2(x)  # 1/8
        x3 = self.layer3(x)  # 1/16
        x4 = self.layer4(x3)  # 1/16
        return x3, x4

    def _init_weight(self):
        for name, m in self.named_modules():
            if 'conv_offset_mask' in name:
                continue
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, layers):
        if layers == [3, 4, 23, 3]:
            pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        elif layers == [3, 4, 6, 3]:
            pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            raise NotImplementedError
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # 参数 name & 参数 size 双重判断
            if k in state_dict and v.size() == state_dict[k].size():
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def build_contextpath_dcn(model, inplanes=64, output_stride=16, pretrained=True, sync_bn=False):
    BatchNorm = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
    if model == 'resnet50':
        return ResNet(Bottleneck, inplanes, output_stride, BatchNorm, pretrained=pretrained,
                      layers=[3, 4, 6, 3], dcn_layers=[0, 4, 6, 3], dcn_interval=1)
    elif model == 'resnet101':
        return ResNet(Bottleneck, inplanes, output_stride, BatchNorm, pretrained=pretrained,
                      layers=[3, 4, 23, 3], dcn_layers=[0, 4, 23, 3], dcn_interval=3)
