# 卫星图语义分割模型使用手册

## 目录结构

```sh
# 数据集
├── datasets
│   ├── base_dataset.py         # 数据集基类
│   ├── build_datasets.py       # 构建 hz 数据集脚本
│   ├── hz
│   │   ├── cls_weights.py      # 样本均衡权重
│   │   ├── hz20.csv            # 原20类颜色
│   │   ├── hz_buildings.csv    # buildings 颜色
│   │   ├── hz_merge.csv        # merge 部分类别合并后颜色
│   │   ├── HZ.py               # hz 数据集类
│   │   ├── mk_dataset.py       # 构建数据集脚本
│   │   └── util.py             # merge 合并类别可用函数
│   ├── __init__.py
│   ├── plt_label_map.py        # 做出各类别的颜色图 
│   └── transforms.py           # 数据增强变换方法

# 测试脚本
├── demo
│   ├── base.py 
│   ├── eval.py     # 评估模型在 trainset, testset 性能; 获取混淆矩阵并作图
│   ├── infer.py    # 模型在整图 or testset 上推理
│   ├── __init__.py
│   └── transfer_npy_to_b.py    # 将模型输出结果的 npy 文件 转成 *.b 文件, 方便 matlab 加载

# 模型训练脚本
├── exps
│   ├── exp.sh   # 命令行执行实验脚本

# 可用模型
├── models
│   ├── bisenet     # 双边语义分割模型
│   │   ├── base.py     # 基本模块
│   │   ├── bisenet.py  # 模型主类
│   │   ├── build_contextpath_dcn.py    # 深层语义分支 + DCN 可变性卷积
│   │   ├── build_contextpath.py        # 深层语义分支
│   ├── deeplab     # deeplab
│   │   ├── aspp.py     # 扩张空间金字塔池化模块
│   │   ├── backbone    # 可用 backbone 模型
│   │   │   ├── drn.py
│   │   │   ├── __init__.py
│   │   │   ├── mobilenet.py
│   │   │   ├── resnet.py
│   │   │   └── xception.py
│   │   ├── decoder.py  # 解码器
│   │   ├── deeplab.py  # 模型主类
│   ├── hrnet       # 高分辨率特征模型 
│   │   ├── cfg         # hrnet 不同结构配置文件
│   │   │   ├── seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
│   │   │   ├── seg_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
│   │   │   ├── seg_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
│   │   │   ├── seg_hrnet_w48_halfblock_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
│   │   │   └── seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
│   │   ├── config.py   # 基本结构
│   │   ├── hrnet.py    # 模型主类结构
│   ├── __init__.py

# 可用模块
├── modules
│   ├── attention   # 注意力机制模块
│   │   ├── __init__.py
│   │   ├── PAM.py  # positional attention module 基于 position 像素位置的注意力机制
│   ├── dcn         # 可变卷积库
│   │   ├── dcn_v2.py
│   │   ├── __init__.py
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── setup.py
│   │   ├── src
│   │   │   ├── cpu
│   │   │   │   ├── dcn_v2_cpu.cpp
│   │   │   │   └── vision.h
│   │   │   ├── cuda
│   │   │   │   ├── dcn_v2_cuda.cu
│   │   │   │   ├── dcn_v2_im2col_cuda.cu
│   │   │   │   ├── dcn_v2_im2col_cuda.h
│   │   │   │   ├── dcn_v2_psroi_pooling_cuda.cu
│   │   │   │   └── vision.h
│   │   │   ├── dcn_v2.h
│   │   │   └── vision.cpp
│   │   └── test.py
│   ├── __init__.py
│   └── sync_batchnorm  # 多 GPU 训练可选用 sync_bn 库
│       ├── batchnorm.py
│       ├── comm.py
│       ├── __init__.py
│       ├── replicate.py
│       └── unittest.py

# hrnet 可用预训练模型
├── pretrained_models
│   ├── hrnetv2_w30_imagenet_pretrained.pth
│   ├── hrnetv2_w48_imagenet_pretrained.pth
│   └── hrnet_w18_small_model_v2.pth

# 训练模型保存目录
├── runs
│   └── HZ_Merge
│       ├── hrnet_ce_Sep10_230343
│       │   ├── checkpoint.pth.tar  # 保存的模型
│       │   ├── events.out.tfevents.1599750225.206.23775.0
│       │   ├── parameters.txt
│       │   ├── Valid_Acc_Acc
│       │   │   └── events.out.tfevents.1599751110.206.23775.2
│       │   ├── Valid_Acc_mAcc
│       │   │   └── events.out.tfevents.1599751110.206.23775.3
│       │   └── Valid_IoU_mIoU
│       │       └── events.out.tfevents.1599751110.206.23775.1

# 可用脚本
├── utils
│   ├── calculate_weights.py    # 计算数据集各类别 balance weight
│   ├── __init__.py
│   ├── loss.py                 # 定义 loss 函数
│   ├── lr_scheduler.py         # 定义 学习率 变换曲线
│   ├── metrics.py              # 语义分割评估方法
│   ├── misc.py                 # 常用的函数
│   ├── saver.py                # 保存训练过程中文件的类
│   ├── trainer.py              # 模型训练的类
│   └── vis.py                  # 可视化相关方法

# 根目录 py 文件
├── argument_parser.py  # 指定命令行可传递的参数
├── constants.py        # 指定常量
├── train.py            # 模型训练入口
```

