from utils import *
from constants import *
from tqdm import tqdm
import random
from constants import hz_dir


def copy_imgs():
    affine_stats = load_pickle(f'{match_dir}/road_affine_stats_num1735_after.pkl')
    for tile in tqdm(affine_stats):
        shutil.copy(
            src=f'{sat16_dir}/{tile}.jpeg',
            dst=f'/datasets/rs_segment/AerialCitys/HZ20/train/image_z16/{tile}.jpeg',
        )


def get_bad_cases():
    bad_case_dir = 'match/bad_cases'
    bad_cases = []
    for txt in os.listdir(bad_case_dir):
        bad_cases += read_txt_as_list(os.path.join(bad_case_dir, txt))
    print('bad case:', len(bad_cases))
    return bad_cases


def split_train_data():
    train_dir = os.path.join(hz_dir, 'train')
    img_dir = os.path.join(train_dir, 'image')
    msk_dir = os.path.join(train_dir, 'mask')

    # train
    # match results
    img_list = [p[:-5] for p in os.listdir(img_dir) if p.endswith('.jpeg')]
    img_list = sorted(set(img_list) - set(get_bad_cases()))  # 1334 = 1735 - 401

    img_paths, target_paths = [], []
    for img in img_list:
        img_paths.append(f'{img_dir}/{img}.jpeg')
        target_paths.append(f"{msk_dir}/{img}.npy")

    # scenes
    scene_img_paths = read_txt_as_list(f'{hz_dir}/train/scenes/img_paths.txt')  # 171
    scene_target_paths = read_txt_as_list(f'{hz_dir}/train/scenes/target_paths.txt')

    # train: 1334 + 171 = 1505
    img_paths += scene_img_paths
    target_paths += scene_target_paths

    write_list_to_txt(img_paths, txt_path=os.path.join(hz_dir, 'train_img_paths.txt'))
    write_list_to_txt(target_paths, txt_path=os.path.join(hz_dir, 'train_target_paths.txt'))
    print('train:', len(img_paths))


def save_valid_data():
    # choose 500 for valid
    img_orders = np.load('datasets/hz/img_orders.npy')
    valid_idxs = img_orders[:300]

    img_paths = read_txt_as_list(os.path.join(hz_dir, 'train_img_paths.txt'))
    target_paths = read_txt_as_list(os.path.join(hz_dir, 'train_target_paths.txt'))

    valid_img_paths, valid_target_paths = [], []
    for i in valid_idxs:
        valid_img_paths.append(img_paths[i])
        valid_target_paths.append(target_paths[i])

    write_list_to_txt(valid_img_paths, f'{hz_dir}/valid_img_paths.txt')
    write_list_to_txt(valid_target_paths, f'{hz_dir}/valid_target_paths.txt')
    print('valid:', len(valid_img_paths))


def save_scene_data():
    scene_dir = os.path.join(hz_dir, 'train/scenes')
    img_dir = os.path.join(scene_dir, 'image')
    msk_dir = os.path.join(scene_dir, 'mask')

    img_list = [p[:-5] for p in os.listdir(img_dir) if p.endswith('.jpeg')]
    img_paths, target_paths = [], []
    for img in img_list:
        img_paths.append(f'{img_dir}/{img}.jpeg')
        target_paths.append(f"{msk_dir}/{img}.npy")

    write_list_to_txt(img_paths, txt_path=os.path.join(scene_dir, 'img_paths.txt'))
    write_list_to_txt(target_paths, txt_path=os.path.join(scene_dir, 'target_paths.txt'))
    print('scenes:', len(img_paths))


def save_test_data():
    root = '/datasets/rs_segment/AerialCitys/HZ20'

    test_dir = os.path.join(root, 'test')
    img_list = ['A', 'B', 'C']

    img_paths, target_paths = [], []
    for img in img_list:
        img_paths.append(f'{test_dir}/{img}18.png')
        target_paths.append(f"{test_dir}/{img}.npy")

    write_list_to_txt(img_paths, f'{root}/test_img_paths.txt')
    write_list_to_txt(target_paths, f'{root}/test_target_paths.txt')
    print('test:', len(img_paths))


def rewrite_file_path():
    from constants import hz_dir

    old_prefix = '/datasets'
    new_prefix = '/home/xs/data'

    def replace_prefix(path_list):
        res_list = []
        for path in tqdm(path_list):
            path = path.replace(old_prefix, new_prefix)
            if os.path.exists(path):
                res_list.append(path)
        return res_list

    txt_list = [
        'train_img_paths.txt', 'train_target_paths.txt',
        'valid_img_paths.txt', 'valid_target_paths.txt',
        'test_img_paths.txt', 'test_target_paths.txt',
    ]

    for txt in txt_list:
        print(txt)
        txt_path = os.path.join(hz_dir, txt)
        train_img_paths = read_txt_as_list(txt_path)
        train_img_paths = replace_prefix(train_img_paths)
        write_list_to_txt(train_img_paths, txt_path)


if __name__ == '__main__':
    # rewrite_file_path()
    # split_trainval_data()
    # save_scene_data()
    save_valid_data()
