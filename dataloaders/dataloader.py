import typing
# from importlib.metadata import FastPath
from typing import Optional, Tuple

from omegaconf import DictConfig
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .EPN_dataloader.epn_control import ControlledEPNDataset
from .patch_learning_dataset import ShapenetDataset
from .patch_learning_dataset import ScannetDataset

def save_iter(dataloader: DataLoader, sampler: Optional[DistributedSampler] = None) -> typing.Iterator:
    """Return a save iterator over the loader, which supports multi-gpu training using a distributed sampler.

    Args:
        dataloader (DataLoader): DataLoader object.
        sampler (Optional[DistributedSampler]): DistributedSampler object.

    Returns:
        typing.Iterator: Iterator object containing data.
    """
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            if sampler is not None:
                sampler.set_epoch(sampler.epoch + 1)
            yield next(iterator)


def get_dataloader(
    opt: DictConfig,phase, random_subsample=False, num_samples=0,select_i=0,rank=0, world_size=1, 
) :
    """
    Return the training and testing dataloaders for the given configuration.

    Args:
        opt (DictConfig): Configuration dictionary.
        sampling (bool): Whether to use sampling.

    Returns:
        Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]: Training and testing dataloaders.
    """
    collate_fn = None

    if opt.mvp_dataset_config.dataset == 'EPN':
        dataset = ControlledEPNDataset( opt,phase=phase)
    elif opt.mvp_dataset_config.dataset == 'shapenet':
        if phase=="train":
            file_name=opt.data.train_file
        elif phase=="val":
            file_name = opt.data.val_novel_file
        elif phase=="test":
            file_name = opt.data.test_file
        dataset = ShapenetDataset( file_name=file_name, data_path=opt.data.data_path, truncation=opt.data.truncation)
    elif opt.mvp_dataset_config.dataset == 'scannet':
        if phase=="train":
            file_name=opt.data.train_file
        elif phase=="val":
            file_name = opt.data.val_novel_file
        elif phase=="test":
            file_name = opt.data.test_file
        dataset = ScannetDataset( file_name=file_name, data_path=opt.data.data_path, truncation=opt.data.truncation,use_bbox=True)
    else:
        raise Exception(opt.mvp_dataset_config.dataset, 'dataset is not supported')

    if opt.distribution_type == "multi":
        sampler = (
            DistributedSampler(dataset, num_replicas=opt.global_size, rank=opt.local_rank)
            if dataset is not None
            else None
        )
    else:
        sampler = None

    # setup the dataloaders
    if phase == 'train':
        dataloader = (
            DataLoader(
                dataset,
                batch_size=opt.training.bs,
                sampler=sampler,
                shuffle=sampler is None,
                num_workers=int(opt.data.workers),
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
            if dataset is not None
            else None
        )
    else:
        dataloader = (
            DataLoader(
                dataset,
                batch_size=opt.evaluation.bs,
                # sampler=sampler,
                shuffle=False, #False
                num_workers=int(opt.data.workers),
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
            if dataset is not None
            else None
        )

    return dataloader, sampler
