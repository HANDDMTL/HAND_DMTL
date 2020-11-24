
from os import path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
sys.path.append(osp.join(cur_dir, '.'))

import torch
from torch.utils import data
try:
    from transform import *
    from constants import NUM_KPS, BONE_HIERARCHY
except:
    from .transform import *
    from .constants import NUM_KPS, BONE_HIERARCHY
    
from os import path as osp
import glob
import copy 
import random
import matplotlib.pyplot as plt
import pickle
import json

class hand_RHD(data.Dataset):
    def __init__(self, dataset_folder, mode='train', crop_size=368, hm_size=[128, 128], pof_size=[128, 128], use_cuda=True, gen_hm=True, gen_pof=False, k_fold=2, sigma=16, *args, **kargs):
        self.mode = mode
        self.samples = []
        self.dataset_folder = dataset_folder
        self.load_dataset(self.dataset_folder)		
        self.base_size, self.crop_size, self.k_fold, self.use_cuda, self.sigma = (crop_size//16 + 1) * 16, crop_size, k_fold, use_cuda, sigma

        if use_cuda:
            gen_hm, gen_pof = False, False

        if self.mode != 'train':
            self.k_fold = 1

        if self.mode=='train':
            self.trans = Compose([
                Crop(size=None, ratio=[1.5, 1.9]),
                ProbOp(op=Rotate(angle=[-30, 30.0], align=False, center=None),p=0.5),
                Resize(s_tg=[self.base_size, self.base_size]),
                Crop(size=[self.crop_size, self.crop_size],ratio=None),
                ProbOp(op=RandomNoise(sigma=0.02), p=0.05),
                ProbOp(op=GaussianBlur(),p=0.1),
                ProbOp(op=ColorJitter(),p=0.1),
                ProbOp(op=GammaAdj(),p=0.3),
                ProbOp(op=ChannelShuffle(),p=0.1),
                ProbOp(op=Gray(),p=0.1),
                ProbOp(op=CutOut(size=[self.crop_size//8, self.crop_size//8]),p=0.5),
                Align(),
                ToTensor(hm_size=hm_size, pof_size=pof_size, sigma=self.sigma,num_kp=NUM_KPS,dist_thresh=24.0, pof_conns=np.array(BONE_HIERARCHY), hm=gen_hm, pof=gen_pof)
            ])
        else:
            self.trans = Compose([
                Crop(size=None, ratio=[1.5, 1.51]),
                Resize(s_tg=[self.crop_size, self.crop_size]),
                Align(),
                ToTensor(hm_size=hm_size, pof_size=pof_size, sigma=self.sigma,num_kp=NUM_KPS,dist_thresh=24.0, pof_conns=np.array(BONE_HIERARCHY),hm=gen_hm, pof=gen_pof)
            ])

        self.hand_pts_index = [0,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17]

    def load_dataset(self, dataset_folder):
        lst_path = osp.join(dataset_folder, 'train.lst' if self.mode=='train' else 'val.lst')
        with open(lst_path, 'r') as fp:
            for s_sp in fp.readlines():
                json_path = s_sp.strip()
                self.samples.append(json_path)

    def __len__(self):
        return int(len(self.samples)//self.k_fold)
    
    def __getitem__(self, index):
        k_fold = max(1, self.k_fold)
        index = index * k_fold + np.random.randint(0, k_fold)
        index = index % len(self.samples)

        lbl_path = osp.join(self.dataset_folder, self.samples[index])
        with open(lbl_path, 'r') as fp:
            lbl = json.load(fp)

        key_points_3d, key_points_2d, img_name, is_left = np.array(lbl['key_points_3d']), np.array(lbl['key_points_2d']), lbl['image_name'].replace('\\', '/'), lbl['is_left']
        if key_points_2d[:, 2].sum() < 8:
            return self.__getitem__(index+1)

        key_points_2d[:, 2] = 1.0-key_points_2d[:, 2]
        image = cv2.imread(osp.join(self.dataset_folder, img_name))
        mask  = cv2.imread(osp.join(self.dataset_folder, img_name.replace('.png', "_mask.png")))

        if not is_left:
            key_points_2d[:, 0] = image.shape[1]-1-key_points_2d[:, 0]
            key_points_3d[:, 0] = -key_points_3d[:, 0]
            image = image[:,::-1].copy()
            mask  = mask[:, ::-1].copy()

        key_points_2d = key_points_2d[self.hand_pts_index]
        key_points_3d = key_points_3d[self.hand_pts_index] * 1000 #m => mm

        visible_flag = np.zeros([NUM_KPS, 1])
        key_points_3d = np.concatenate([key_points_3d[...,:3], visible_flag], 1)

        mano_theta, mano_beta = np.zeros([46]), np.zeros([11])

        R = {
            'dt_name'        : 'hand_RHD',
            'image'          : image,
            'seg_mask'       : mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'     : mano_theta,
            'mano_beta'      : mano_beta,
        }

        return self.trans(R)