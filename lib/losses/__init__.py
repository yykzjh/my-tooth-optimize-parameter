import torch
import torch.nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

import configs.config as config

from lib.losses.DiceLoss import DiceLoss


loss_function_list = ['DiceLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss', 'MSELoss', 'SmoothL1Loss', 'L1Loss',
                      'WeightedSmoothL1Loss', 'BCEDiceLoss', 'BCEWithLogitsLoss']


def create_loss(device=None):

    if config.loss_function_name == 'DiceLoss':
        return DiceLoss(classes=config.classes, weight=torch.tensor(config.class_weight, device=device),
                        sigmoid_normalization=config.sigmoid_normalization, skip_index_after=config.skip_index_after)

    else:
        raise RuntimeError(f"Unsupported loss function: '{config.loss_function_name}'. Supported losses: {loss_function_list}")




class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)
