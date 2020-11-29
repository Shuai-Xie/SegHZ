from datasets.hz.HZ import HZ_Merge
from utils.vis import plt_img_target
from utils.misc import recover_color_img
from constants import hz_dir


def build_hz_datasets(base_size, crop_size, merge_all_buildings=False):
    trainset = HZ_Merge(hz_dir, 'train', base_size, crop_size, merge_all_buildings)
    validset = HZ_Merge(hz_dir, 'valid', base_size, crop_size, merge_all_buildings)
    testset = HZ_Merge(hz_dir, 'test', base_size, crop_size, merge_all_buildings)

    return trainset, validset, testset


def vis_data(trainset):
    for idx, sample in enumerate(trainset):
        img, target = sample['img'], sample['target']
        img, target = img.squeeze(0).numpy(), target.squeeze(0).numpy()
        img = recover_color_img(img)
        target = trainset.remap_fn(target.astype('uint8'))
        plt_img_target(img, target, trainset.label_colors)


if __name__ == '__main__':
    trainset, validset, testset = build_hz_datasets(base_size=1024,
                                                    crop_size=512,
                                                    merge_all_buildings=True)
    vis_data(trainset)
    # vis_data(validset)
    # vis_data(testset)
