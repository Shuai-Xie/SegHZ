import argparse


def parse_args(params=None):
    parser = argparse.ArgumentParser(description="AerialCitys")

    # model
    parser.add_argument('--seg_model', type=str, default='deeplab')
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet18', 'resnet50', 'resnet101', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='Cityscapes')
    parser.add_argument('--hrnet-width', type=int, default=48)

    # input size
    parser.add_argument('--base-size', type=int, default=513, help='base image size')
    parser.add_argument('--crop-size', type=int, default=513, help='crop image size')

    # loss
    parser.add_argument('--loss-type', type=str, default='ce', help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iters-per-epoch', type=int, default=None, help='iterations per epoch')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: True)')
    parser.add_argument('--merge-all-buildings', action='store_true', default=False,
                        help='whether to use balanced weights (default: True)')

    # optimizer params
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default='False', help='whether use nesterov (default: False)')

    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'cos', 'patience'],
                        help='lr scheduler mode: (default: poly)')

    # seed
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    # checking point
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume model path')

    args = parser.parse_args(params)

    return args
