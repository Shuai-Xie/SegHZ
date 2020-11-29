# HRNet 代码说明

- Github: https://github.com/HRNet/HRNet-Semantic-Segmentation
- Paper: https://arxiv.org/abs/1908.07919



### HRNet 结构

HRNet 主要的模型结构，具体实现部分在 `HighResolutionNet` 类中有详细定义。

总体结构 按照顺序 可分为三部分：

1. **stem net：**

   - 从 IMG 到 1/4 大小的 feature map，得到此尺寸的特征图后，之后的 HRNet feature map 始终保持此尺寸

2. **HRNet 4 stages：**如下图所示的 4 阶段 由 `HighResolutionModule` 组成的模型

   - 其中，每个蓝色底色为1个阶段
   - 每个 stage 产生的 multi-scale 特征图，具体配置如下表，以  hrnet_48 为例
   - stage 的连接处有 transition 结构，用于在不同 stage 之间连接，完成 channels 及 feature map 大小对应

   |        | multi-scale feature map |  num_branches<br /> (分支数) | num_blocks<br />(每个分支 block 重复次数) | num_modules<br />(HighResolutionModule 重复次数) |
   | :----: | :---------------------: | :-------------------------: | :---------------------------------------- | ------------------------------------------------------------ |
   | stage1 |          [1/4]          |               1              | [4]                                       | 0                                                            |
   | stage2 |       [1/4, 1/8]        |               2              | [4, 4]                                    | 1                                                            |
   | stage3 |    [1/4, 1/8, 1/16]     |               3              | [4, 4, 4]                                 | 4                                                            |
   | stage4 | [1/4, 1/8, 1/16, 1/32]  |               4              | [4, 4, 4, 4]                              | 3                                                            |

![image-20201002194911116](https://upload-images.jianshu.io/upload_images/1877813-5f7899585a868b0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3. **segment head：**
   - 将 stage4 输出的 4 种 scale 特征 concat 到一起
   - 加上 `num_channels -> num_classes` 层，得到分割结果



### HRNet 构建函数 `def HRNet(cfg_path, **kwargs)`

1. 通过指定 `cfg_path` 选择要使用的模型的结构（yaml 存储）
2. 通过指定 kwargs 选择是否选用 pretrain 模型

具备 pretrain 模型的，可用模型结构：

- seg_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
- seg_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
- seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml，为目前采用的结构

```python
def HRNet(cfg_path, **kwargs):
    from models.hrnet.config import update_config

    cfg = update_config(cfg_path)
    model = HighResolutionNet(cfg, **kwargs)
    if kwargs.get('use_pretrain', False):
        model.load_pretrain(cfg.MODEL.PRETRAINED)

    return model
```

yaml 文件中，关于模型结构的关键部分，以 hrnet_w48 为例

```yaml
MODEL:
  NAME: seg_hrnet
  ALIGN_CORNERS: True
  PRETRAINED: 'pretrained_models/hrnetv2_w48_imagenet_pretrained.pth'  # 指定 pretrain 模型路径
  EXTRA:  # EXTRA 具体定义了模型的结果，包括 4 个 STAGE，各自的参数
    FINAL_CONV_KERNEL: 1
    STAGE1:
      NUM_MODULES: 1
      NUM_RANCHES: 1
      BLOCK: BOTTLENECK
      NUM_BLOCKS:
      - 4
      NUM_CHANNELS:
      - 64
      FUSE_METHOD: SUM
    STAGE2:
      NUM_MODULES: 1	# HighResolutionModule 重复次数
      NUM_BRANCHES: 2   # 分支数
      BLOCK: BASIC
      NUM_BLOCKS:
      - 4
      - 4
      NUM_CHANNELS:
      - 48
      - 96
      FUSE_METHOD: SUM
    STAGE3:
      NUM_MODULES: 4
      NUM_BRANCHES: 3
      BLOCK: BASIC
      NUM_BLOCKS:
      - 4
      - 4
      - 4
      NUM_CHANNELS:
      - 48
      - 96
      - 192
      FUSE_METHOD: SUM
    STAGE4:
      NUM_MODULES: 3
      NUM_BRANCHES: 4
      BLOCK: BASIC
      NUM_BLOCKS:
      - 4
      - 4
      - 4
      - 4
      NUM_CHANNELS:
      - 48
      - 96
      - 192
      - 384
      FUSE_METHOD: SUM
```



### HRNet 类 `class HighResolutionNet(nn.Module)`

#### 1. 结构初始化 `__init__()`

HRNet 类定义，通过 config 指定的模型结构，实例化特定结构的模型，构建过程如下

```python
def __init__(self, config, **kwargs):
    """
    # stem net
    # 两层 3x3 conv，stride=2，得到 1/4 大小的 feature map
    
    # 开始 HRModule 阶段
    # 每个 stage 不仅保留之前所有 size 的特征，还增加一个新的下采样 size 特征
    # stage1: [1/4]
    # stage2: [1/4, 1/8]
    # stage3: [1/4, 1/8, 1/16]
    # stage4: [1/4, 1/8, 1/16, 1/32]

    # last_layers，即 segment head
    # 从 num_channels 到 num_classes，完成语义分割
    """
```



#### 2. 构建 stage 间转换层 `_make_transition_layer()`

transition layer 完成 stage 之间连接需要的 两种转换
- input channels 转换
- feature size downsample

```python
def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
    """
        :param num_channels_pre_layer: pre_stage output channels list
        :param num_channels_cur_layer: cur_stage output channels list
            cur 总比 pre 多一个 output_channel 对应增加的 1/2 下采样
                    stage2      stage3          stage4
            pre:    [256]       [48,96]         [48,96,192]
            cur:    [48,96]     [48,96,192]     [48,96,192,384]

            每个 stage channels 数量也对应了 stage2/3/4 使用 BASIC block; expansion=1
            使用 BASIC block 考虑到本身 feature map 数量并不像 resnet 那么大
        :return:
            transition_layers:
                1.完成 pre_layer 到 cur_layer input channels 数量对应
                2.完成 feature map 尺寸对应
        """
```

以下为 hrnet_w48 的 transition 具体结构，ModuleList 定义并行 module 结构

```python
# stage 1-2
  (transition1): ModuleList(
    # input channels，从 1/4 到 1/4，完成通道数量转换
    (0): Sequential(
      (0): Conv2d(256, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    # input channels + downsample，从 1/4 到 1/8，不仅通道数量，而且使用 stride=2 进行下采样
    (1): Sequential(
      (0): Sequential(
        (0): Conv2d(256, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
 
# stage 2-3
  (transition2): ModuleList(
    (0): None  # 因为 同层对应的连接处的 feature map channels 和 size 一致，所以不需要转换
    (1): None
    # downsample，stage2 末尾，从 1/8 到 1/16，需要使用 stride=2 下采样
    (2): Sequential(
      (0): Sequential(
        (0): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
  
# stage 3-4
  (transition3): ModuleList(
    (0): None
    (1): None
    (2): None
    # downsample，同 stage2 用法一样，因为前3个branch对应的 feature map 可以直接连接，所以只要对末尾完成 1/16 到 1/32 下采样
    (3): Sequential(
      (0): Sequential(
        (0): Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
```



#### 3. 构建 stage1 的 layer `_make_layer()`

stage1 产生 1/4 feature map，没有 branch 分支结构，采用与 resnet 完成一样的 `_make_layer()` 函数构建层

```python
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
            nn.Conv2d(inplanes, planes * block.expansion,  # expansion = 4
                      kernel_size=1, stride=stride, bias=False),
            BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)
```



#### 4. 构建 stage 2/3/4 的 layer `_make_stage`

stage 2/3/4 为 HRNet 核心结构，用到了 `HighResolutionModule`，内含 branch 构建和 特征 fuse 模块

```python
def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
    """
        创建 num_modules 个 HighResolutionModule 结构，
        	每个 HighResolutionModule 末尾完成所有分支的特征融合，使得高分辨率特征也编码尽可能多的深层语义特征
        :param layer_config: 从 yaml config 文件读取到的 stage 配置
        :param num_inchannels: 由 NUM_CHANNELS 和 block.expansion 相乘得到
        :param multi_scale_output: 都是 True
        :return:
            num_modules 个 HighResolutionModule 串联结构
            其中每个 HighResolutionModule 先有 branch 分支并行，末尾处再将不同 scale 的特征交叉 sum 融合
        """
    # eg. stage2
    num_modules = layer_config['NUM_MODULES']  # 1, HighResolutionModule 重复次数
    num_branches = layer_config['NUM_BRANCHES']  # 2, 并行分支数，高度
    num_blocks = layer_config['NUM_BLOCKS']  # [4,4]，每个分支 block 重复次数
    num_channels = layer_config['NUM_CHANNELS']  # [48,96]，每个分支 channels
    block = blocks_dict[layer_config['BLOCK']]  # BASIC
    fuse_method = layer_config['FUSE_METHOD']  # SUM，multi scale 特征融合方式

    modules = []
    for i in range(num_modules):  # HighResolutionModule 重复次数
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
```



### HRNet 核心模块类 `class HighResolutionModule(nn.Module)`

实现下图红框中的，branch 并行 多 scale 特征提取 和 末端将 多 scale 特征通过 upsample/downsample 方式融合

![image-20201002204544176](https://upload-images.jianshu.io/upload_images/1877813-d118629786b09e81.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
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
```



#### 构建一个横向分支 `_make_one_branch()`

上图红框中，每个横向的串行结构，如第1个红框 stage2 内，有2个横向的串行结构；由 `num_blocks` 决定串行 `block` 使用个数

```python
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
```



#### 构建 HighResolutionModule 多个并行的 branches `_make_branches()`

根据 stage cfg 中指定的 branch 数量，构建多个并行的 branch，调用之前的 `_make_one_branch()`，如 stage 2/3/4 各有 2/3/4 个 branches

```python
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
```



#### 构建 multi-scale 特征融合层 `_make_fuse_layers()`

HighResolutionModule 末尾的特征融合层

以下图红框即 stage3 中 蓝色 branch 输出结果为例，其输出结果要转换成 4 种尺度的特征，用于每个 branch 末尾的特征融合

- 1/8 ↗ 1/4，不同层，channel 不同，size 不同 👉 通道转换 + 上采样 (在 forward 函数中由双线性插值完成)
- 1/8 → 1/8，相同层，channel 一致，size 一致 👉 None，直接使用 feature
- 1/8 ↘ 1/16，不同层，channel 不同，size 不同 👉 通道转换 + 下采样 (通过串联的 stride=2 的 3x3 conv 完成)
- 1/8 ↘ 1/32，同上

![image-20201002205717387](https://upload-images.jianshu.io/upload_images/1877813-62b893d5b31ee251.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
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
```

