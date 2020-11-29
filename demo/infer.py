import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from PIL import Image
from torch.utils.data import DataLoader
from pprint import pprint
import numpy as np
import cv2

from datasets.build_datasets import build_hz_datasets
from datasets.hz.helper import get_merge_func
from datasets.transforms import get_img_transfrom

from demo.base import *
from models import *
from utils import *
from utils.metrics import Evaluator
from constants import *

dataset = 'HZ_Merge'

num_classes = 7
base_size, crop_size = 1024, 1024
merge_all_buildings = True

label_names, label_colors = get_label_name_colors(f'datasets/hz/hz{num_classes}.csv')

# exp = 'hz7_limit_cos_infer1024_Oct06_201243'
exp = 'hrnet64_epoch100_bs8_limitcos_validinfer1024_Nov03_170817'


# load model
def load_model():
    model = HRNet(
        cfg_path='models/hrnet/cfg/seg_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        num_classes=num_classes, use_pretrain=False
    )
    load_state_dict(model, f'runs/{dataset}/{exp}/checkpoint.pth.tar')
    model.eval().cuda()
    return model


def demo_validset(train=False):
    trainset, validset, _ = build_hz_datasets(base_size, crop_size, merge_all_buildings)
    save_dir = f'data/valid/{exp}'
    mkdir(save_dir)

    if train:
        trainset.transform = validset.transform
        validset = trainset
        save_dir = f'data/train/{exp}'

    mkdir(save_dir)

    torch.manual_seed(100)
    dataloader = DataLoader(validset, batch_size=1, shuffle=True)
    model = load_model()

    for idx, sample in enumerate(dataloader):
        img, target = sample['img'].cuda(), sample['target']
        img_name = os.path.basename(sample['path'][0]).split('.')[0]  # 随 batch_size 变为 list

        # 因为 sliding 所以引入误差？
        pred = predict_sliding(model, img, num_classes, crop_size)
        pred = to_numpy(pred, toint=True)

        target = target.squeeze(0).numpy()
        target = cv2.resize(target, pred.shape[::-1], interpolation=cv2.INTER_NEAREST)
        acc = np.sum(pred == target) / target.size
        title = "acc_{:0.3f}_{}".format(acc, img_name)
        print(idx, title)

        img = recover_color_img(img.squeeze(0))
        target = validset.remap_fn(target)
        plt_img_target_pred(img, target, pred + 1, label_colors,
                            vis=True,
                            save_path=f'{save_dir}/{title}.png',
                            title=title)


def demo_imgs(save_result=True):
    base_dir = os.path.join(hz_dir, 'test')
    save_dir = f'data/test/{exp}'
    save_dir = save_dir.replace('1024', '512')
    mkdir(save_dir)

    merge_fn = get_merge_func(merge_all_buildings)  # map target to merge idxs

    test_trans = get_img_transfrom()
    evaluator = Evaluator(num_classes)

    model = load_model()

    imgs = [img for img in os.listdir(base_dir) if not img.endswith('.npy') and img != '@eaDir']
    # imgs = ['A18.png', 'B18.png', 'C18.png']
    # imgs = ['airport18.png', 'highway18.png', 'jianggan18.png', 'river18.png', 'xiaoshan18.png']
    result = {}

    crop_size = 512
    for img in imgs:
        print(img)
        img_name = img.replace('18.png', '').replace('.jpeg', '')
        ori_img = Image.open(f'{base_dir}/{img}').convert('RGB')

        # trans to infer
        img = test_trans(ori_img).unsqueeze(0).cuda()
        target = np.load(f'{base_dir}/{img_name}.npy').astype('uint8')
        target = merge_fn(target)

        pred = predict_sliding(model, img,
                               num_classes, crop_size,
                               overlap=0.25, return_pred=True)
        pred = to_numpy(pred, toint=True)
        # pred = cv2.resize(pred, target.shape[::-1], interpolation=cv2.INTER_NEAREST)  # reshape target/pred 效果差不多
        pred += 1  # remap to 1-C

        target = cv2.resize(target, pred.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # cal acc
        valid_mask = target != 0
        acc = np.sum(pred[valid_mask] == target[valid_mask]) / valid_mask.sum()
        print(img_name, 'acc:', acc)

        # save pred
        if save_result:
            pred_c = color_code_target(pred, label_colors)
            cv2.imwrite(f'{save_dir}/{img_name}.png', pred_c[:, :, ::-1])

        # error
        error_mask = np.zeros_like(target)
        mask = pred != target
        error_mask[mask] = target[mask]
        plt_img_target_pred_error(ori_img, target, pred, error_mask, label_colors,
                                  dpi=100,
                                  save_path=f'{save_dir}/{img_name}_err.png' if save_result else None,
                                  title='acc: %.3f' % acc)

        evaluator.reset()
        evaluator.add_batch(target - 1, pred - 1)
        data_eval = {
            'Acc': evaluator.Pixel_Accuracy(),
            'accs': evaluator.Pixel_Accuracy_Class(),
            'ious': evaluator.Intersection_over_Union_Class(),
            'mIoU': evaluator.Mean_Intersection_over_Union(),
            'fwIoU': evaluator.Frequency_Weighted_Intersection_over_Union(),
        }
        pprint(data_eval)
        result[img_name] = {
            'pred': pred,
            'eval': data_eval
        }

    # dump_pickle(result, f'{save_dir}/result.pkl')


def save_GT_mask():
    base_dir = os.path.join(hz_dir, 'test')
    save_dir = 'data/GT'

    imgs = [img for img in os.listdir(base_dir) if not img.endswith('.npy') and img != '@eaDir']

    merge_fn = get_merge_func(merge_all_buildings)  # map target to merge idxs

    for img in imgs:
        print(img)
        img_name = img.replace('18.png', '').replace('.jpeg', '')

        target = np.load(f'{base_dir}/{img_name}.npy').astype('uint8')
        target = merge_fn(target)
        target_c = color_code_target(target, label_colors)
        cv2.imwrite(f'{save_dir}/{img_name}.png', target_c[:, :, ::-1])


if __name__ == '__main__':
    # save_GT_mask()
    demo_imgs()
    # demo_validset(train=True)
