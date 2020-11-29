import os
import sys

sys.path.insert(0, '/nfs/xs/tmp/SegHZ')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from pprint import pprint
from tqdm import tqdm

from datasets.build_datasets import build_hz_datasets
from demo.base import *
from models import *
from utils import *
from utils.metrics import Evaluator

dataset = 'HZ_Merge'

num_classes = 7
base_size, crop_size = 1024, 1024
merge_all_buildings = True

label_names, label_colors = get_label_name_colors(f'datasets/hz/hz{num_classes}.csv')

exp = 'hz7_limit_cos_infer1024_Oct06_201243'


# load model
def load_model():
    # model = BiSeNet(num_classes, context_path='resnet101', use_dcn=False)
    model = HRNet('models/hrnet/cfg/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                  num_classes=num_classes, use_pretrain=False)
    load_state_dict(model, f'runs/{dataset}/{exp}/checkpoint.pth.tar')
    model.eval().cuda()
    return model


def eval_validset(train=False):
    """
    class IoU & mIoU, Acc & mAcc
    """
    trainset, validset, _ = build_hz_datasets(base_size, crop_size, merge_all_buildings)
    eval_path = f'runs/{dataset}/{exp}/valid_eval_crop{crop_size}.pkl'
    cm_path = f'runs/{dataset}/{exp}/valid_confusion_matrix_crop{crop_size}.npy'

    if train:
        trainset.transform = validset.transform
        validset = trainset
        eval_path = f'runs/{dataset}/{exp}/train_eval_crop{crop_size}.pkl'
        cm_path = f'runs/{dataset}/{exp}/train_confusion_matrix_crop{crop_size}.npy'

    dataloader = DataLoader(validset, batch_size=4, shuffle=False)

    model = load_model()
    evaluator = Evaluator(num_classes)
    evaluator.reset()

    for sample in tqdm(dataloader):
        image, target = sample['img'].cuda(), sample['target']

        pred = predict_sliding(model, image, num_classes, crop_size)
        pred, target = to_numpy(pred), to_numpy(target)
        evaluator.add_batch(target, pred)

    # save confusion matrix
    np.save(cm_path, evaluator.confusion_matrix)

    data_eval = {
        'Acc': evaluator.Pixel_Accuracy(),
        'accs': evaluator.Pixel_Accuracy_Class(),
        'ious': evaluator.Intersection_over_Union_Class(),
        'mIoU': evaluator.Mean_Intersection_over_Union(),
        'fwIoU': evaluator.Frequency_Weighted_Intersection_over_Union(),
    }
    dump_pickle(data_eval, eval_path)
    pprint(data_eval)


def data_cls_eval():
    res = load_pickle(f'runs/{dataset}/{exp}/valid_eval.pkl')
    plt_class_evals(res, label_colors)


def data_confusion():
    cm = np.load(f'runs/{dataset}/{exp}/valid_confusion_matrix.npy')
    plt_confusion_matrix(cm, label_names)


if __name__ == '__main__':
    eval_validset(train=True)
    pass
