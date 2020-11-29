import torch
import constants
from modules.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from utils.loss import *
from utils.self_operation import SelfOperation
from utils import *
from tqdm import tqdm
from demo.base import *
import datasets.hz.config as config


class Trainer:

    def __init__(self, args, model, train_set, valid_set, test_set, saver, writer):
        self.args = args
        self.saver = saver
        self.saver.save_experiment_config()  # save cfgs
        self.writer = writer

        self.num_classes = train_set.num_classes

        # dataloaders
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_set.make_dataset_multiple_of_batchsize(args.batch_size)
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, **kwargs)  # 尺寸相同，可并行
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)  # 尺寸不同每次1张大图
        self.dataset_size = {'train': len(train_set), 'valid': len(valid_set), 'test': len(test_set)}
        print('dataset size:', self.dataset_size)

        self.label_names = train_set.label_names

        # iters_per_epoch
        self.iters_per_epoch = args.iters_per_epoch if args.iters_per_epoch else len(self.train_loader)

        # optimizer & lr_scheduler
        train_params = [{'params': model.parameters(), 'lr': args.lr}]  # 默认会有 pam 参数
        self.optimizer = torch.optim.SGD(train_params,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay,
                                         nesterov=args.nesterov)
        self.lr_scheduler = LR_Scheduler(mode=args.lr_scheduler,  # 逐渐减小的 lr
                                         base_lr=args.lr,
                                         num_epochs=args.epochs,
                                         iters_per_epoch=self.iters_per_epoch)

        # model = torch.nn.DataParallel(model)
        self.model = model.cuda()

        # loss
        cls_weight = None
        if args.use_balanced_weights:
            cls_weight = torch.from_numpy(config.new_1505['weights'])
            print(cls_weight)

        if args.loss_type == 'ce':
            self.criterion = CELoss(weight=cls_weight, multi_weights=args.multi_weights)
        elif args.loss_type == 'ce_thre':
            self.criterion = Thre_CELoss(weight=cls_weight, multi_weights=args.multi_weights)
            # loss 噪声抑制, p=0.05 在 epoch=70 左右达到, 忽略模型认为 95% 概率出错的
            self.loss_thre = LossThre(min_p=0.01, max_p=0.1, epochs=args.epochs)
        elif args.loss_type == 'ce_ohem':
            self.criterion = OHEM_CELoss(weight=cls_weight, multi_weights=args.multi_weights)
        elif args.loss_type == 'label_smooth':
            self.criterion = LabelSmoothing_CELoss(weight=cls_weight, lbl_smooth=0.1)
        else:
            raise NotImplementedError

        self.criterion.cuda()

        # evaluator
        self.evaluator = Evaluator(self.num_classes)
        self.best_Acc = 0.0
        self.best_fwIoU = 0.0

        self.with_aux_loss = len(self.args.multi_weights) > 1

        # self_op
        self.op = SelfOperation(opt=None)

    def training(self, epoch, prefix='Train'):
        self.model.train()
        self.criterion.train()

        train_losses = AverageMeter()
        if self.with_aux_loss:
            aux_losses = [AverageMeter() for _ in range(len(self.args.multi_weights))]

        tbar = tqdm(self.train_loader, total=self.iters_per_epoch)

        # update loss_thre，高于 thre 的认为是噪声
        if self.args.loss_type == 'ce_thre':
            l_thre = self.loss_thre(epoch)
            self.criterion.set_loss_thre(l_thre)  # cosine reduce
            self.writer.add_scalar(f'{prefix}/loss_thre', l_thre, epoch)

        for i, sample in enumerate(tbar):
            if i == self.iters_per_epoch:  # 执行完到 tbar 更新到最后，再跳出
                break

            # update lr each iteration
            self.lr_scheduler(self.optimizer, i, epoch - 1)  # epoch 从1开始

            image, target = sample['img'].cuda(non_blocking=True), sample['target'].cuda(non_blocking=True)

            self.optimizer.zero_grad()

            output = self.model(image)
            loss, losses = self.criterion(output, target)

            # self_op
            if self.args.with_self_supervise:
                self_image = self.op(image.clone())
                self_output = self.op(self.model(self_image), inv=True)
                loss += self.criterion(self_output, target)[0]  # 数据增强 loss
                loss += ((output - self_output) ** 2).mean()  # 旋转匹配 mse loss

            loss.backward()

            if self.args.seg_model == 'hrnet_pam':  # 防止学成 NaN
                nn.utils.clip_grad_value_(self.model.pam[3].gamma, clip_value=1.0)  # (min=-clip_value, max=clip_value)

            self.optimizer.step()

            train_losses.update(loss.item())

            disp = 'Epoch {}, Train loss: {:.3f}'.format(epoch, train_losses.avg)

            if self.with_aux_loss:
                for i in range(len(aux_losses)):
                    aux_losses[i].update(losses[i].item())
                    disp += ', aug_{}: {:.3f}'.format(i, aux_losses[i].avg)
            tbar.set_description(disp)

        self.writer.add_scalar(f'{prefix}/lr', get_learning_rate(self.optimizer), epoch)
        self.writer.add_scalars(f'{prefix}/loss', {
            'total': train_losses.val
        }, epoch)
        if self.with_aux_loss:
            self.writer.add_scalars(f'{prefix}/loss', {
                f'aux_{i}': aux_losses[i].val for i in range(len(aux_losses))
            }, epoch)

        if self.args.seg_model == 'hrnet_pam':
            self.writer.add_scalar(f'{prefix}/gamma', self.model.pam[3].gamma.item(), epoch)

    @torch.no_grad()
    def validation(self, epoch, test=False):
        self.model.eval()
        self.criterion.eval()
        self.evaluator.reset()  # reset confusion matrix

        if test:
            tbar, prefix = tqdm(self.test_loader), 'Test'
        else:
            tbar, prefix = tqdm(self.valid_loader), 'Valid'

        # loss
        segment_losses = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['img'].cuda(), sample['target'].cuda()
            output = predict_sliding(self.model, image,
                                     self.num_classes, self.args.base_size,  # base_size: 1024 valid 整图推理
                                     overlap=0.25, return_pred=False)
            segment_loss = self.criterion(output, target)  # segment
            segment_losses.update(segment_loss.item())
            tbar.set_description(f'{prefix} loss: %.4f' % segment_losses.avg)

            # eval, add result
            pred = torch.argmax(output, dim=1)
            self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W

        Acc = self.evaluator.Pixel_Accuracy()

        accs = self.evaluator.Pixel_Accuracy_Class()
        mAcc = self.evaluator.Mean_Pixel_Accuracy(accs)

        ious = self.evaluator.Intersection_over_Union_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union(ious)
        fwIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union(ious)

        print('Epoch: {}, Acc:{:.4f}, mAcc:{:.4f}, mIoU:{:.4f}, FWIoU:{:.4f}'.format(
            epoch, Acc, mAcc, mIoU, fwIoU))

        self.writer.add_scalar(f'{prefix}/loss', segment_losses.avg, epoch)

        # global iou/Acc
        self.writer.add_scalars(f'{prefix}/IoU', {
            'mIoU': mIoU,
            'FWIoU': fwIoU,
        }, epoch)
        self.writer.add_scalars(f'{prefix}/Acc', {
            'mAcc': mAcc,
            'Acc': Acc,
        }, epoch)

        # cls acc/iou
        self.writer.add_scalars(f'{prefix}/iou_cls', {
            '{:0>2d}_{}'.format(i + 1, self.label_names[i + 1]): ious[i] for i in range(self.num_classes)
        }, epoch)
        self.writer.add_scalars(f'{prefix}/acc_cls', {
            '{:0>2d}_{}'.format(i + 1, self.label_names[i + 1]): accs[i] for i in range(self.num_classes)
        }, epoch)

        if not test and fwIoU > self.best_fwIoU:  # iou 评估
            # if Acc > self.best_pixelAcc:  # Acc 评估
            print('saving model...')
            self.best_Acc = Acc
            self.best_fwIoU = fwIoU

            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                'Acc': Acc,
                'mAcc': mAcc,
                'mIoU': mIoU,
                'fwIoU': fwIoU,
                'accs': accs,
                'ious': ious,
            }
            self.saver.save_checkpoint(state)
            print('save model at epoch', epoch)

        if test:
            self.saver.save_test_results({
                'epoch': epoch,
                'Acc': Acc,
                'mAcc': mAcc,
                'mIoU': mIoU,
                'fwIoU': fwIoU,
                'accs': accs,
                'ious': ious,
            })
            print('save test results')

        return fwIoU, Acc

    def load_best_checkpoint(self, file_path=None, load_optimizer=False):
        checkpoint = self.saver.load_checkpoint(file_path=file_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if file_path:
            print('load', file_path)
        print(f'=> loaded checkpoint - epoch {checkpoint["epoch"]}')
        return checkpoint["epoch"]
