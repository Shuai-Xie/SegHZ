import numpy as np
from tqdm import tqdm
from utils import entropy


def calculate_class_weights(dataloader, num_classes, save_path=None):
    z = np.zeros((num_classes,))
    tqdm_batch = tqdm(dataloader)
    tqdm_batch.set_description('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['target']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()

    # freqs
    total_frequency = np.sum(z)
    freqs = z / total_frequency

    # weights
    class_weights = []
    for freq in freqs:
        w = 1 / (np.log(1.02 + freq))
        class_weights.append(w)
    class_weights = np.array(class_weights)

    if save_path:
        np.save(save_path, class_weights)

    stats = {
        'cnts': z,
        'freqs': freqs,
        'weights': class_weights
    }
    from pprint import pprint
    pprint(stats)
    print('entro:', entropy(freqs))

    return class_weights


def cmp_old_new_cls_weights():
    from datasets.hz.HZ import HZ_Merge
    from datasets.transforms import get_transform
    from torch.utils.data import DataLoader

    dset_roots = [
        # '/datasets/rs_segment/AerialCitys/HZ20/old_dataset',
        '/datasets/rs_segment/AerialCitys/HZ20'
    ]
    num_classes = 7
    for root in dset_roots:
        dset = HZ_Merge(root, 'train', 1024, 512, merge_all_buildings=True)
        dset.transform = get_transform('test', base_size=1024)

        dataloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)
        calculate_class_weights(dataloader, num_classes)
