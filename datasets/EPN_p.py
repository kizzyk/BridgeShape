import torch
import numpy as np
import os.path as osp
import torch.utils.data as data
from loguru import logger
def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines

class ControlledEPNDataset_change(data.Dataset):
    def __init__(self,
                 config,
                 phase='train',
                 input_transform=None, #None
                 target_transform=None, #None
                 augment_data=False):

        self.data_root = config.data_root
        if config.per_class:
            data_paths = read_txt(osp.join(self.data_root, 'splits', phase+'_'+config.class_id+'.txt'))
            logger.success('Loading {} data from class {}'.format(phase,config.class_id))
        else:
            data_paths = read_txt(osp.join(self.data_root, 'splits', phase+'.txt'))
        data_paths = [data_path for data_path in data_paths]
        self.config = config
        self.representation = config.representation
        self.trunc_thres = config.trunc_thres
        self.log_df = config.log_df
        self.data_paths = data_paths
        self.augment_data = augment_data
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.suffix = config.suffix
        print(phase," data len : ",len(self.data_paths))
        print(phase," trunc_thres : ",self.trunc_thres)

    def load(self, filename):
        return torch.load(filename)

    def name(self):
        return 'ControlledEPNDataset'

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        filename_partial = osp.join(self.data_root , self.data_paths[index])
        filename = osp.join(self.data_root ,"shapenet_dim64_df_8c_npy", self.data_paths[index])
        scan_id = osp.basename(filename).replace(self.suffix, '')
        input_sdf = self.load(filename_partial)[0]
        gt_file = filename.replace(self.suffix, '')[:-3] + '0__.npy'
        gt_df = torch.from_numpy(np.load(gt_file))

        if self.representation == 'tsdf':
            #print(input_sdf.shape)
            #print(input_sdf.dtype)
            #print(gt_df.shape)
            #print(gt_df.dtype)
            #print(input_sdf)
            input_sdf = np.clip(input_sdf, -self.trunc_thres, self.trunc_thres)
            #print(input_sdf)
            gt_df = np.clip(gt_df, 0.0, self.trunc_thres)
            #print(gt_df)

        if self.log_df:
            gt_df = np.log(gt_df + 1)

        # Transformation
        if self.input_transform is not None:
            input_sdf = self.input_transform(input_sdf)
        if self.target_transform is not None:
            gt_df = self.target_transform(gt_df)

        return scan_id,input_sdf, gt_df
