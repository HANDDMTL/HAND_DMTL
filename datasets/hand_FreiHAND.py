
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

class hand_FreiHAND(data.Dataset):
    def __init__(self, dataset_folder, mode='train', crop_size=368, hm_size=[128, 128], pof_size=[128, 128], use_cuda=True, gen_hm=True, gen_pof=False, k_fold=1, sigma=16, *args, **kargs):
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
                Crop(size=None, ratio=[1.5, 1.51]),
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

    def load_dataset(self, dataset_folder):
        camera_K_path, joints_path = osp.join(dataset_folder, 'training_K.json'), osp.join(dataset_folder, 'training_xyz.json')
        
        camera_ks, joints = np.array(json.load(open(camera_K_path, 'r'))), np.array(json.load(open(joints_path, 'r')))

        for (camera_k, joint) in zip(camera_ks, joints):
            self.samples.append([camera_k, joint])

    def __len__(self):
        return int(len(self.samples)//self.k_fold)
    
    def project3d(self, joints, K):
        uv = np.matmul(K, joints.T).T
        return uv[:, :2] / uv[:, -1:]

    def __getitem__(self, index):
        k_fold = max(1, self.k_fold)
        index  = index * k_fold + np.random.randint(0, k_fold)
        index  = index % len(self.samples)

        (camera_k, key_points_3d) = copy.deepcopy(self.samples[index])

        if self.mode=='train':
            img_path = osp.join(self.dataset_folder, 'training/rgb',  str(index+len(self.samples)*np.random.randint(0, 4)*0).zfill(8)+'.jpg')
        else:
            img_path = osp.join(self.dataset_folder, 'training/rgb',  str(index).zfill(8)+'.jpg')
            
        msk_path = osp.join(self.dataset_folder, 'training/mask', str(index).zfill(8)+'.jpg')

        image, mask = cv2.imread(img_path), cv2.imread(msk_path)[...,0]
        
        #flip the right hand to left hand
        key_points_3d[:, 0] = -key_points_3d[:, 0]
        image, mask = image[:,::-1].copy(), mask[:, ::-1].copy()
        key_points_2d = self.project3d(key_points_3d, camera_k)
        
        visible_flag = np.zeros([NUM_KPS, 1])
        key_points_3d = np.concatenate([key_points_3d*1000, visible_flag], 1) #m to mm
        key_points_2d = np.concatenate([key_points_2d, visible_flag], 1)
        
        mano_theta, mano_beta = np.zeros([46]), np.zeros([11])

        R = {
            'dt_name'        : 'FreiHAND',
            'image'          : image,
            'seg_mask'       : mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'     : mano_theta,
            'mano_beta'      : mano_beta,
        }

        return self.trans(R)