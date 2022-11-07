from lib.losses.AbstractDiceLoss import _AbstractDiceLoss
from lib.losses.utils import *


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=1, weight=None, sigmoid_normalization=True, skip_index_after=None):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, epsilon=1e-6, weight=self.weight)
