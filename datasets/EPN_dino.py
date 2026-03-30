
import os.path as osp
import torch.utils.data as data
from loguru import logger
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def process_depth_and_normal(depth_image, normal_image):
    
    depth_image = np.array(depth_image).astype(np.float32)

    
    depth_min, depth_max = depth_image.min(), depth_image.max()
    depth_image = (depth_image - depth_min) / (depth_max - depth_min)
    depth_image = np.stack([depth_image, depth_image, depth_image], axis=-1)

    normal_image = np.array(normal_image).astype(np.float32)
    normal_image = normal_image / 255  
    normal_image = normal_image[:, :, :3] * normal_image[:, :, 3:]
    return depth_image, normal_image

def load_data_from_files(intrinsic_file_path, extrinsic_file_path, subfolder_path,transform,scale_offset_matrix_path):

    scale_path = os.path.join(subfolder_path, scale_offset_matrix_path)
    scale=np.loadtxt(scale_path)[0]
    depth_imgs = []
    normal_imgs = []
    intrinsics = []
    extrinsics = []
    for i in range(6):
        depth_path = os.path.join(subfolder_path, f"depth_{i}.png")
        normal_path = os.path.join(subfolder_path, f"normals_{i}.png")
        intrinsic_path = os.path.join(subfolder_path,f"{i}_"+ intrinsic_file_path)
        extrinsic_path = os.path.join(subfolder_path,f"{i}_"+ extrinsic_file_path)
        intrinsics.append(np.loadtxt(intrinsic_path))
        extrinsics.append(np.loadtxt(extrinsic_path))
        if os.path.exists(depth_path) and os.path.exists(normal_path):
            depth_img = np.array(Image.open(depth_path).resize((518, 518), Image.Resampling.NEAREST)).astype(np.float32)
            normal_img = np.array(Image.open(normal_path).resize((518, 518), Image.Resampling.LANCZOS)).astype(np.float32)
            depth_img, normal_img = process_depth_and_normal(depth_img, normal_img)

            depth_imgs.append(depth_img)
            normal_imgs.append(normal_img)

    assert len(depth_imgs) == 6 and len(normal_imgs) == 6
    depth_imgs=np.stack(depth_imgs, axis=0) # [6, H, W ,3]
    normal_imgs=np.stack(normal_imgs, axis=0)  # [6, H, W ,3]
    intrinsics=np.stack(intrinsics, axis=0)  # [6, 3 ,3]
    extrinsics=np.stack(extrinsics, axis=0)  # [6, 3 ,4]

    depth_imgs = torch.from_numpy(depth_imgs).permute(0, 3, 1, 2).float()  # [6, 3, H, W]
    normal_imgs = torch.from_numpy(normal_imgs).permute(0, 3, 1, 2).float()  # [6, 3, H, W]
    depth_imgs = transform(depth_imgs)
    normal_imgs = transform(normal_imgs)


    intrinsics = torch.from_numpy(intrinsics).float()  # [6, 3, 3]
    extrinsics = torch.from_numpy(extrinsics).float()  # [6, 3, 4]
    scale=torch.tensor(scale)
    # 返回加载的所有数据
    return depth_imgs, normal_imgs, intrinsics, extrinsics,scale


class ControlledEPNDataset_dino(data.Dataset):
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

        self.intrinsic_file_path = 'K.txt'
        self.extrinsic_file_path = 'RT.txt'
        self.scale_offset_matrix_path = 'scale_offset_matrix.txt'
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
        dinov2_path=osp.join(self.data_root ,"shapenet_dim64_df_8c_npy","depth_normal", self.data_paths[index][:-9]+'gt')#,"dinov2.npy")
        depth_imgs, normal_imgs, intrinsics, extrinsics,scale=\
            load_data_from_files(self.intrinsic_file_path, self.extrinsic_file_path, dinov2_path, self.transform,self.scale_offset_matrix_path)
        if self.representation == 'tsdf':
            input_sdf = np.clip(input_sdf, -self.trunc_thres, self.trunc_thres)
            gt_df = np.clip(gt_df, 0.0, self.trunc_thres)

        if self.log_df:
            gt_df = np.log(gt_df + 1)

        # Transformation
        if self.input_transform is not None:
            input_sdf = self.input_transform(input_sdf)
        if self.target_transform is not None:
            gt_df = self.target_transform(gt_df)

        return scan_id,input_sdf, gt_df, depth_imgs, normal_imgs, intrinsics, extrinsics,scale
