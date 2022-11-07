import random

import numpy as np

from .elastic_deform import ElasticTransform
from .random_crop import RandomCropToLabels
from .random_flip import RandomFlip
from .random_rescale import RandomRescale
from .random_rotate import RandomRotation
from .random_shift import RandomShift
from .gaussian_noise import GaussianNoise

functions = ['elastic_deform', 'gaussian_noise', 'random_crop', 'random_flip', 'random_rescale',
             'random_rotate', 'random_shift']


class RandomChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensor, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensor, label
        t = random.choice(self.transforms)

        return t(img_tensor, label)


class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensor, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensor, label

        for t in self.transforms:
            img_tensor, label = t(img_tensor, label)

        return img_tensor, label
