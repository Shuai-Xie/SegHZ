from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
from pprint import pprint


# https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter
def read_test_results(exp='runs/HZ_Merge/hrnet_w48_epoch50_bs4_ce_Nov26_202816'):
    dirs = sorted(os.listdir(exp))

    test_results = {
        'Acc': [],
        'IoU': [],
        'acc': [],
        'iou': []
    }
    for dir in dirs:
        if 'Test' not in dir:
            continue
        sub_dir = os.path.join(exp, dir)

        if os.path.isdir(os.path.join(sub_dir)):
            ea = event_accumulator.EventAccumulator(path=sub_dir)
            ea.Reload()

            for scalar_key in ea.scalars.Keys():
                vals = [it.value for it in ea.scalars.Items(scalar_key)]
                if len(vals) == 1:
                    vals = vals[0]
                for res_key in test_results:
                    if res_key in scalar_key:
                        test_results[res_key].append(vals)
                        break

    pprint(test_results)


def runs_dir_tests():
    runs_dir = 'runs/HZ_Merge'

    for exp in os.listdir(runs_dir):
        print(exp)
        read_test_results(exp=os.path.join(runs_dir, exp))


if __name__ == '__main__':
    # read_test_results()
    runs_dir_tests()
