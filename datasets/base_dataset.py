
import numpy as np
# from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    dataset = None

    # decide resolution later at model
    if opt.dataset_mode == 'ControlledEPNDataset':
        from .EPN_p import ControlledEPNDataset_change
        train_dataset = ControlledEPNDataset_change(opt,phase='train')
        test_dataset = ControlledEPNDataset_change(opt,phase='test')
    elif opt.dataset_mode == 'ControlledEPNDataset_32':
        from .EPN import ControlledEPNDataset_32
        train_dataset = ControlledEPNDataset_32(opt,phase='train')
        test_dataset = ControlledEPNDataset_32(opt,phase='test')
    elif opt.dataset_mode == 'ControlledEPNDataset_dino':
        from .EPN_dino import ControlledEPNDataset_dino
        train_dataset = ControlledEPNDataset_dino(opt,phase='train')
        test_dataset = ControlledEPNDataset_dino(opt,phase='test')
    elif opt.dataset_mode == 'ControlledEPNDataset_128':
        from .EPN_128 import ControlledEPNDataset_128
        train_dataset = ControlledEPNDataset_128(opt,phase='train')
        test_dataset = ControlledEPNDataset_128(opt,phase='test')
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset
