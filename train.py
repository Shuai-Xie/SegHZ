import argument_parser
from pprint import pprint

args = argument_parser.parse_args()
pprint(vars(args))

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.tensorboard import SummaryWriter
from datasets.build_datasets import build_hz_datasets
from models import *
from utils.saver import Saver
from utils.trainer import Trainer
from utils.misc import get_curtime
import torch


def is_interval(epoch):
    return epoch % args.eval_interval == (args.eval_interval - 1)


def get_eval_interval_old(cur_epoch, num_epochs):
    if cur_epoch >= num_epochs * 0.9:
        return 1
    elif cur_epoch >= num_epochs * 0.8:
        return 2
    elif cur_epoch >= num_epochs * 0.6:
        return 5
    else:
        return num_epochs


def get_eval_interval(cur_epoch, num_epochs):
    if cur_epoch >= num_epochs * 0.75:
        return 1
    elif cur_epoch >= num_epochs * 0.5:
        return 2
    else:
        return 5


def load_aux_pretrain(model):
    ckpt = torch.load('runs/HZ_Merge/hrnet_w64_epoch50_bs4_ce_Nov26_203227/checkpoint.pth.tar')
    pretrained_dict = ckpt['state_dict']
    model_dict = model.state_dict()
    new_pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys() and 'cls_head' not in k}  # ocr 压缩 head 960->512
    aux_head = {
        # conv
        'aux_head.0.weight': pretrained_dict['final_conv.0.weight'],
        # 'aux_head.0.bias': pretrained_dict['final_conv.0.bias'], # 没设置
        # bn
        'aux_head.1.weight': pretrained_dict['final_conv.1.weight'],
        'aux_head.1.bias': pretrained_dict['final_conv.1.bias'],
        'aux_head.1.running_mean': pretrained_dict['final_conv.1.running_mean'],
        'aux_head.1.running_var': pretrained_dict['final_conv.1.running_var'],
        'aux_head.1.num_batches_tracked': pretrained_dict['final_conv.1.num_batches_tracked'],
        'aux_head.3.weight': pretrained_dict['cls_head.weight'],  # 需要这部分 减少 aug0 loss
        'aux_head.3.bias': pretrained_dict['cls_head.bias'],
    }
    # size 是匹配的
    # for key in aux_head:
    #     print(model_dict[key].shape, aux_head[key].shape)
    new_pretrained_dict.update(aux_head)  # add aux_head param
    model_dict.update(new_pretrained_dict)  # replace same key param
    model.load_state_dict(model_dict)
    print('load pretrain')


def load_pretrain(model):
    ckpt = torch.load('runs/HZ_Merge/hrnet_w64_epoch50_bs4_ce_Nov26_203227/checkpoint.pth.tar')
    pretrained_dict = ckpt['state_dict']
    model_dict = model.state_dict()
    new_pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys() and 'cls_head' not in k}  # 默认 head 重新学
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    print('load pretrain')


def get_model(num_classes):
    if args.seg_model == 'hrnet':
        model = HRNet(f'models/hrnet/cfg/seg_hrnet_w{args.hrnet_width}.yaml', num_classes, use_pretrain=True)
        args.multi_weights = [1.]

    # aux 结构，保证 hrnet 基础结构 stage features 性能
    # 在此基础上 concat new features: PSP, PAM
    elif args.seg_model == 'hrnet_psp':
        model = HRNet_PSP(f'models/hrnet/cfg/seg_hrnet_w{args.hrnet_width}.yaml', num_classes, use_pretrain=False, with_aux=False)
        # load_aux_pretrain(model)
        # args.multi_weights = [0.8, 1.]
        load_pretrain(model)
        args.multi_weights = [1.]

    elif args.seg_model == 'hrnet_pam':
        model = HRNet_PAM(f'models/hrnet/cfg/seg_hrnet_w{args.hrnet_width}.yaml', num_classes, use_pretrain=False)
        load_aux_pretrain(model)
        args.multi_weights = [0.4, 1.]

    elif args.seg_model == 'hrnet_ocr':
        # 直接从头训练 OCR 是有问题的，在之前的 hrnet64 基础上，用那个作为 pretrain
        model = HRNet_OCR(f'models/hrnet/cfg/seg_hrnet_ocr_w{args.hrnet_width}.yaml', num_classes, use_pretrain=False)
        load_aux_pretrain(model)
        args.multi_weights = [0.4, 1.]

    else:
        raise NotImplementedError

    return model


def main():
    # dataset
    trainset, validset, testset = build_hz_datasets(args.base_size, args.crop_size, args.merge_all_buildings)
    # model
    model = get_model(num_classes=trainset.num_classes)

    args.checkname = f'{args.seg_model}_noaux_w{args.hrnet_width}_epoch{args.epochs}_bs{args.batch_size}_{args.loss_type}'

    saver = Saver(args, timestamp=get_curtime())
    writer = SummaryWriter(saver.experiment_dir)
    trainer = Trainer(args, model, trainset, validset, testset, saver, writer)

    for epoch in range(1, args.epochs + 1):
        trainer.training(epoch)
        if epoch % get_eval_interval(epoch, args.epochs) == 0:
            trainer.validation(epoch)
    print('Valid ===> best fwIoU:', trainer.best_fwIoU, 'pixelAcc:', trainer.best_Acc)

    # test
    epoch = trainer.load_best_checkpoint()
    test_fwIoU, test_pixelAcc = trainer.validation(epoch, test=True)
    print('Test ===> best fwIoU:', test_fwIoU, 'pixelAcc:', test_pixelAcc)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
