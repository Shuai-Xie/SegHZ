import numpy as np

# https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier
epsilon = 1e-5


def calculate_miou(confusion_matrix):
    MIoU = np.divide(np.diag(confusion_matrix), (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix)))
    MIoU = np.nanmean(MIoU)
    return MIoU


def fast_hist(a, b, n):
    """
    a and b are predict and mask respectively
    n is the number of classes
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


class Evaluator(object):
    """
    confusion matrix
       Pred
    GT  1 ... n
        2
       ...
        n
    """

    def __init__(self, num_class):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # TP / ALL pixels
        return Acc

    def Pixel_Accuracy_Class(self):
        accs = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return accs

    def Mean_Pixel_Accuracy(self, accs=None):
        accs = self.Pixel_Accuracy_Class() if accs is None else accs
        mAcc = np.nanmean(accs)
        return mAcc

    def Intersection_over_Union_Class(self):
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return iu

    def Mean_Intersection_over_Union(self, iu=None):
        iu = self.Intersection_over_Union_Class() if iu is None else iu
        MIoU = np.nanmean(iu)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self, iu=None):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self.Intersection_over_Union_Class() if iu is None else iu
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Mean_Dice(self):
        inter = np.diag(self.confusion_matrix)  # vector
        dices = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        dice = np.divide(2 * inter, dices)
        dice = np.nanmean(dice)
        return dice

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]  # i:gt, j:pre
        count = np.bincount(label, minlength=self.num_class ** 2)  # total classes on confusion_matrix
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, return_miou=False):
        assert gt_image.shape == pre_image.shape  # np img, B,H,W
        confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += confusion_matrix
        if return_miou:
            return calculate_miou(confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def dump_matrix(self, path):
        np.save(path, self.confusion_matrix)
