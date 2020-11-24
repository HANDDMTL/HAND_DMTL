
from os import path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
sys.path.append(osp.join(cur_dir, '.'))

import numpy as np
import os
import cv2
import random
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from PIL import Image

'''
    R:
        image:		   h x w x c represented as np.array with np.uint8 format
        seg_mask:      h x w, which gives the segmentation mask with np.uint8 format
        key_points_2d: 21 x 3	 3	<=========> [x, y, conf], where x, y in image space by pixel
        key_points_3d: 21 x 4	 4	<=========> [x, y, z, conf], in which x, y and z are represented by mm.
        mano_theta:    46		 46 <=========> [{rx, ry, rz} x 15 + is_exist]
        mano_beta:	   11		 11 <=========> [10 + is_exist]

        2d visible:
            0:  in image,      visible,         annotated
            1:  in image,      not visible,     annotated
            2:  not in image,  not visible,     annotated
            3:  out of image,  not visible,     not annotated
            4:  -,             -，              not annotated
        
        3d visible:
               0:  annotated
               1:  not annotated
    
    marks: we take the left hand coordinate system for 3d key points
              z
             /
            /
           /
          /
         /
        o-----------> x
        |
        |
        |
        |
        |
        v
        y
'''

# n x 3 
def bbox(key_points_2d, min_kp=2):
    valid = key_points_2d[:, 2] < 2
    Lbl = key_points_2d[valid, :2]
    return (None, None) if Lbl.shape[0]<min_kp else (np.min(Lbl, 0).astype(np.int), np.max(Lbl, 0).astype(np.int))

def filter_key_points_2d(h, w, key_points_2d):
    for l in range(len(key_points_2d)):
        x, y =  key_points_2d[l, :2]
        if x < 0 or y < 0 or x >= w or y >= h:
            if key_points_2d[l, 2] < 2:
                key_points_2d[l, 2] = 2
    return key_points_2d

class Resize(object):
    def __init__(self, s_tg, *args, **kargs):
        self.s_tg = s_tg

    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']
        seg_mask      = R['seg_mask']

        sh, sw, c = image.shape
        th, tw = int(self.s_tg[1]), int(self.s_tg[0])
        image = cv2.resize(image, (tw,th), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        fh, fw = th/sh,tw/sw
        f = np.array([fw,fh]).reshape(-1, 2).astype(np.float)
        key_points_2d[:,:2] = key_points_2d[:, :2] * f

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class GammaAdj(object):
    def __init__(self, sf=[0.7,1.3],iso=True, *args, **kargs):
        self.sf, self.iso = sf, iso
    
    def __call__(self, R):
        def gamma_adj_impl__(I, gamma):
            return ((I.astype(np.float) / 255.0)**gamma)*255.0
        if self.iso:
            f = random.uniform(self.sf[0], self.sf[1])
        else:
            fr = random.uniform(self.sf[0], self.sf[1])
            fg = random.uniform(self.sf[0], self.sf[1])
            fb = random.uniform(self.sf[0], self.sf[1])

        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']
        seg_mask      = R['seg_mask']

        if self.iso:
            image = gamma_adj_impl__(image, f).copy().astype(np.uint8)
        else:
            B, G, R = image[...,0], image[...,1], image[...,2]
            B, G, R = gamma_adj_impl__(B, fb), gamma_adj_impl__(G, fg), gamma_adj_impl__(R, fr)
            image = np.stack((B,G,R),2).copy().astype(np.uint8)
        
        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class GaussianBlur(object):
    def __init__(self, k=[1,3,5,7], s=[0,1,2,3,4], *args, **kargs):
        self.k, self.s = k, s

    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']

        k, s  = np.random.choice(self.k), np.random.choice(self.s)
        image = cv2.GaussianBlur(image, (k,k), s).astype(np.uint8).copy()

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class Gray(object):
    def __init__(self, *args, **kargs):
        pass

    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = np.repeat(image[...,np.newaxis], 3, 2).astype(np.uint8).copy()

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class ChannelShuffle(object):
    def __init__(self, *args, **kargs):
        pass
    
    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']

        b,g,r=np.random.choice([0,1,2],3,replace=False)
        image = image[...,[b,g,r]].copy()

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class ColorJitter(object):
    def __init__(self, b=0.1, c=0.1, s=0.15, h=0.2, *args, **kargs):
        self.b, self.c, self.s, self.h = b, c, s, h
        self.jitter = torchvision.transforms.ColorJitter(b,c,s,h)

    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']
        
        PImg = Image.fromarray(image)
        PImg = self.jitter(PImg)
        image = np.asarray(PImg).astype(np.uint8)

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class RandomNoise(object):
    def __init__(self, mode='gaussian', sigma=0.05, *args, **kargs):
        self.mode, self.sigma = mode, sigma
    
    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']

        if self.mode == 'gaussian':
            eps = np.random.normal(0,1,size=image.shape)*self.sigma*255.0
            image = np.clip(image+eps, 0, 255).astype(np.uint8)

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class Crop(object):
    def __init__(self, size=[224,224], ratio=[1.1, 1.4], *args, **kargs):
        assert size is None or ratio is None
        self.size, self.ratio  = size, ratio

    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']

        sh, sw = image.shape[:2]
        if self.size is not None: #crop a fixed size
            th, tw = self.size[1], self.size[1]
            assert th <= sh and tw <=sw
            dx = random.randint(0,sw-tw)
            dy = random.randint(0,sh-th)
            off = np.array([dx,dy]).reshape(-1, 2)
            image = image[dy:dy+th,dx:dx+tw].astype(np.uint8)
            seg_mask = seg_mask[dy:dy+th,dx:dx+tw].astype(np.uint8)
            key_points_2d[:, :2] -= off
        else: #crop with 2d pose key points
            r = random.uniform(self.ratio[0], self.ratio[1])
            lt, rb = bbox(key_points_2d)
            expand = (rb-lt).max() / 2.0 * r
            center = ((lt + rb) / 2.0).astype(np.int)
            expand = np.array([expand, expand]).astype(np.int)
            lt, rb = center - expand, center + expand
            h, w, c = image.shape
            DImg = np.zeros((rb[1]-lt[1], rb[0]-lt[0], c)).astype(np.uint8)
            s_lt = np.clip(lt, [0, 0], [w-1,h-1])
            s_rb = np.clip(rb, [0, 0], [w-1,h-1])
            r_lt = np.clip(-lt, a_min=[0, 0], a_max=None)
            dx, dy = s_rb - s_lt
            DImg[r_lt[1]:r_lt[1]+dy, r_lt[0]:r_lt[0]+dx] = image[s_lt[1]:s_rb[1], s_lt[0]:s_rb[0]]
            image = DImg.astype(np.uint8)
            mask = np.zeros((rb[1]-lt[1], rb[0]-lt[0])).astype(np.uint8)
            mask[r_lt[1]:r_lt[1]+dy, r_lt[0]:r_lt[0]+dx] = seg_mask[s_lt[1]:s_rb[1], s_lt[0]:s_rb[0]]
            seg_mask = mask.astype(np.uint8)
            key_points_2d[:, :2] -= lt

        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }
    

class CutOut(object):
    def __init__(self, size=[20, 20], color=None, *args, **kargs):
        self.size, self.color = size, color

    def __call__(self, R, *args, **kargs):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']

        sh, sw = image.shape[:2]
        th, tw = self.size[1], self.size[1]

        dx = random.randint(0,sw-tw)
        dy = random.randint(0,sh-th)
        
        image[dy:dy+th, dx:dx+tw] = self.color if self.color is not None else np.random.randint(0, 255)
        
        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        } 
        
class Align(object):
    def __init__(self):
        self._register_key_points_2d_weight_()
        self._register_mask_weight_()

    def _register_mask_weight_(self):
        self.w_mask = {
            'cmu_hand':0.0,
            'FreiHAND':1.0,
            'HIU':1.0,
            'hand_RHD':1.0,
            'hand_STB':0.0,
            'hand_dexter':0,0,
        }

        self.w_ssl = { #enable self-supervised learning or not.
            'cmu_hand':1.0,
            'FreiHAND':0.0,
            'HIU':0.0,
            'hand_RHD':0.0,
            'hand_STB':0.0,
            'hand_dexter':1,0,
        }

        self.w_pof = { #enable the pof or not
            'cmu_hand':0.0,
            'FreiHAND':1.0,
            'HIU':0.0,
            'hand_RHD':0.0,
            'hand_STB':1.0,
            'hand_dexter':0,0,
        }

        self.w_hm = { #enable the heat-map of not
            'cmu_hand':0.0,
            'FreiHAND':1.0,
            'HIU':1.0,
            'hand_RHD':0.0,
            'hand_STB':0.0,
            'hand_dexter':0,0,
        }
    
    def _register_key_points_2d_weight_(self):
        self.w_key_points_2d = {
            'cmu_hand'    :  np.array([2.5, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7])*3.0,
            'FreiHAND'    :  np.array([2.5, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7])*1.0,
            'HIU'         :  np.array([2.5, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7])*3.0,
            'hand_RHD'    :  np.array([2.5, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7])*1.0,
            'hand_STB'    :  np.array([2.5, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7])*1.0,
            'hand_dexter' :  np.array([2.5, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7, 1.7,1,1,1.7])*1.0,
        }

        self.w_key_points_3d = {
            'cmu_hand'    :  np.array([1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0])*1.0,
            'FreiHAND'    :  np.array([1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0])*1.0,
            'HIU'         :  np.array([1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0])*1.0,
            'hand_RHD'    :  np.array([1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0])*1.0,
            'hand_STB'    :  np.array([1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0])*1.0,
            'hand_dexter' :  np.array([1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0])*1.0,
        }


    def __call__(self, R, *args, **kargs):
        dt_name = R['dt_name']
        R['w_mask']            = self.w_mask[dt_name]
        R['w_key_points_2d']   = self.w_key_points_2d[dt_name]
        R['w_key_points_3d']   = self.w_key_points_3d[dt_name]
        R['w_ssl']             = self.w_ssl[dt_name]
        R['w_pof']             = self.w_pof[dt_name]
        R['w_hm']              = self.w_hm[dt_name]
        return R

class Rotate(object):
    def __init__(self,	angle=[-30,30], center=None, align=True, *args, **kargs):
        self.angle, self.center, self.align = angle, center, align

    def __call__(self, R):
        dt_name		  = R['dt_name']
        image		  = R['image']
        seg_mask      = R['seg_mask']
        key_points_2d = R['key_points_2d']
        key_points_3d = R['key_points_3d']
        mano_theta	  = R['mano_theta']
        mano_beta	  = R['mano_beta']

        if self.center=='Image':
            h, w = image.shape[:2]
            center=((w-1)/2.0,(h-1)/2.0)
        else:
            lt, rb = bbox(key_points_2d)
            center = (lt + rb) / 2.0 if self.center is None else self.center

        angle = random.uniform(self.angle[0], self.angle[1])
        M = cv2.getRotationMatrix2D((center[0], center[1]),angle,math.cos((angle%45)*3.1415926/180.0))
        h, w = image.shape[:2]
        image = cv2.warpAffine(image,M,(w,h)).astype(np.uint8)

        seg_mask = (cv2.warpAffine(seg_mask, M, (w, h)) > 128).astype(np.uint8) * 255

        R_mat, R_trans = M.T[:2], M.T[2:3]
        key_points_2d[:, :2] = np.dot(key_points_2d[:, :2], R_mat) + R_trans
        key_points_3d[:, :2] = np.dot(key_points_3d[:, :2], R_mat)
    
        return {
            'dt_name'		 : dt_name,
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
        }

class Compose(object):
    def __init__(self, ops, *args, **kargs):
        self.ops = ops

    def __call__(self, R):
        for op in self.ops:
            R = op(R)
        return R

class POF_Generator(object):
    def __init__(self, pof_size=[368, 368], pof_conns=np.array([[0, 1], [1, 2]]), dist_thresh=24.0, normalize=True, exploit_uvz=True, k=100, *args, **kargs):
        dist_thresh = max(pof_size[1], pof_size[0]) / 368 * dist_thresh
        self.num_pof, self.pof_size, self.pof_conns, self.dist_thresh=len(pof_conns),pof_size, pof_conns, dist_thresh
        self.x_m       = np.tile(np.arange(pof_size[0]), (self.num_pof, pof_size[1], 1))
        self.y_m       = np.tile(np.arange(pof_size[1]), (self.num_pof, pof_size[0], 1)).transpose([0, 2, 1])
        self.scale     = np.array(pof_size).reshape(1, 2)
        self.normalize = normalize
        self.k = k
        self.exploit_uvz = exploit_uvz

    def __call__(self, key_points_2d, key_points_3d, *args, **kargs):
        pt_xy, pt_valid_2d, pt_xyz, pt_valid_3d = key_points_2d[:, :2]*self.scale, key_points_2d[:, 2], key_points_3d[:, :3], key_points_3d[:, 3]<1
        num_pof, pof_size, pof_conns = self.num_pof, self.pof_size, self.pof_conns
        pt_a_2d, pt_b_2d = pt_xy[pof_conns[:, 0]], pt_xy[pof_conns[:, 1]]
        vab = pt_b_2d - pt_a_2d
        vab = vab / (np.sqrt(np.square(vab).sum(-1)).reshape(-1, 1) + 1e-6)
        dx, dy = self.x_m - pt_a_2d[:, 0].reshape(-1, 1, 1), self.y_m-pt_a_2d[:, 1].reshape(-1, 1, 1)
        dist = np.abs(dy*vab[:, 0].reshape(-1, 1, 1)-dx*vab[:, 1].reshape(-1, 1, 1))

        x_min, x_max = np.minimum(pt_a_2d[:, 0], pt_b_2d[:, 0]) - self.dist_thresh, np.maximum(pt_a_2d[:, 0], pt_b_2d[:, 0]) + self.dist_thresh
        y_min, y_max = np.minimum(pt_a_2d[:, 1], pt_b_2d[:, 1]) - self.dist_thresh, np.maximum(pt_a_2d[:, 1], pt_b_2d[:, 1]) + self.dist_thresh
        x_min, x_max, y_min, y_max = x_min.reshape(-1, 1, 1), x_max.reshape(-1, 1, 1), y_min.reshape(-1, 1, 1), y_max.reshape(-1, 1, 1)

        if not self.exploit_uvz:
            mask = ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[pof_conns[:,0]]<3) * (pt_valid_2d[pof_conns[:,1]]<3) * pt_valid_3d[pof_conns[:,0]] * pt_valid_3d[pof_conns[:,0]]).reshape(-1, 1, 1)).astype(np.float)
        else:
            mask =  ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[pof_conns[:,0]]<3) * (pt_valid_2d[pof_conns[:,1]]<3)).reshape(-1, 1, 1)).astype(np.float)

        pt_a_3d, pt_b_3d = pt_xyz[pof_conns[:, 0]], pt_xyz[pof_conns[:, 1]]
        vab_3d = pt_b_3d - pt_a_3d
        if self.normalize:
            vab_3d = vab_3d / (np.sqrt(np.square(vab_3d).sum(-1)).reshape(-1, 1) +1e-6)

        if not self.exploit_uvz:
            pof_value = vab_3d
        else:
            pof_value = np.concatenate([vab, vab_3d[:,2:3]], 1)

        pof = np.repeat(mask[:, :, :, np.newaxis], 3, axis=3) * pof_value.reshape(-1, 1, 1, 3)
        pof = pof.transpose(0, 3, 1, 2).reshape(-1, pof_size[1], pof_size[0])

        pt_valid_2d = ((pt_valid_2d[pof_conns[:,0]]<3) * (pt_valid_2d[pof_conns[:,1]]<3)).reshape(-1, 1, 1).astype(np.float)
        pt_valid_3d = (pt_valid_3d[pof_conns[:, 0]] * pt_valid_3d[pof_conns[:, 1]]).reshape(-1, 1, 1).astype(np.float)

        if not self.exploit_uvz:
            pof_valid = mask*pt_valid_3d*pt_valid_2d + pt_valid_3d*pt_valid_2d
            pof_valid = pof_valid[..., np.newaxis].repeat(3, axis=3).transpose(0, 3, 1, 2).reshape(-1, pof_size[1], pof_size[0]).astype(np.float)
        else:
            pof_valid_a = np.tile(pt_valid_2d.reshape(-1, 1, 1, 1), (1, pof_size[1], pof_size[0], 2))
            pof_valid_b = np.tile(pt_valid_3d.reshape(-1, 1, 1, 1), (1, pof_size[1], pof_size[0], 1))
            pof_valid = np.concatenate([pof_valid_a, pof_valid_b], 3).transpose(0, 3, 1, 2).reshape(-1, pof_size[1], pof_size[0]).astype(np.float)

        return np.nan_to_num(pof), pof_valid

class POF_Generator_cuda(nn.Module):
    def __init__(self,pof_size=[368, 368], pof_conns=np.array([[0, 1], [1, 2]]), k=100, dist_thresh=24.0, normalize=True, exploit_uvz=True, *args, **kargs):
        super(POF_Generator_cuda, self).__init__()
        dist_thresh = max(pof_size[1], pof_size[0]) / 368 * dist_thresh
        self.num_pof, self.pof_size = pof_conns.shape[0], pof_size
        x_m = np.tile(np.arange(pof_size[0]), (self.num_pof, pof_size[1], 1)).reshape(1, self.num_pof, pof_size[1], pof_size[0])
        y_m = np.tile(np.arange(pof_size[1]), (self.num_pof, pof_size[0], 1)).transpose([0, 2, 1]).reshape(1, self.num_pof, pof_size[1], pof_size[0])
        self.register_buffer('x_m',			torch.from_numpy(x_m).float())
        self.register_buffer('y_m',			torch.from_numpy(y_m).float())
        self.register_buffer('dist_thresh', torch.tensor(dist_thresh).float())
        self.register_buffer('pof_conns',	torch.from_numpy(pof_conns).long())
        self.register_buffer('scale',		torch.tensor(pof_size).float().reshape(1, 1, 2))
        self.normalize = normalize
        self.k = k
        self.exploit_uvz=exploit_uvz

    #key_points_2d batch x k x 3
    def forward(self, key_points_2d, key_points_3d, *args, **kargs):
        nb, nk = key_points_2d.shape[0:2]
        pt_xy, pt_valid_2d, pt_xyz, pt_valid_3d = key_points_2d[..., :2]*self.scale, key_points_2d[..., 2], key_points_3d[..., :3], key_points_3d[..., 3] < 1
        num_pof, pof_size, pof_conns = self.num_pof, self.pof_size, self.pof_conns
        pt_a_2d, pt_b_2d = pt_xy[:, pof_conns[:, 0]], pt_xy[:, pof_conns[:, 1]]
        vab = F.normalize(pt_b_2d - pt_a_2d, p=2, dim=2) # n x k x 2
        dx, dy = self.x_m - pt_a_2d[..., 0].reshape(nb, num_pof, 1, 1), self.y_m-pt_a_2d[..., 1].reshape(nb, num_pof, 1, 1)
        dist = torch.abs(dy*vab[..., 0].reshape(nb, num_pof, 1, 1)-dx*vab[..., 1].reshape(nb, num_pof, 1, 1))

        pt_x = torch.stack([pt_a_2d[...,0], pt_b_2d[...,0]], -1)
        pt_y = torch.stack([pt_a_2d[...,1], pt_b_2d[...,1]], -1)
        x_min, x_max = pt_x.min(-1)[0].reshape(nb, num_pof, 1, 1)-self.dist_thresh, pt_x.max(-1)[0].reshape(nb, num_pof, 1, 1)+self.dist_thresh
        y_min, y_max = pt_y.min(-1)[0].reshape(nb, num_pof, 1, 1)-self.dist_thresh, pt_y.max(-1)[0].reshape(nb, num_pof, 1, 1)+self.dist_thresh

        if not self.exploit_uvz:
            mask = ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[:, pof_conns[:, 0]]<3) * (pt_valid_2d[:, pof_conns[:, 1]]<3) * pt_valid_3d[:, pof_conns[:, 0]] * pt_valid_3d[:, pof_conns[:, 1]]).reshape(nb, -1, 1, 1)).float()
        else:
            mask = ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[:, pof_conns[:, 0]]<3) * (pt_valid_2d[:, pof_conns[:, 1]]<3)).reshape(nb, -1, 1, 1)).float()

        if self.normalize:
            vab_3d = F.normalize(input=pt_xyz[:, pof_conns[:, 1]]-pt_xyz[:, pof_conns[:, 0]], p=2, dim=2)
        else:
            vab_3d = pt_xyz[:, pof_conns[:, 1]]-pt_xyz[:, pof_conns[:, 0]]

        if not self.exploit_uvz:
            pof_value = vab_3d
        else:
            pof_value = torch.cat([vab, vab_3d[:,:,2:3]], 2)

        pof = mask.reshape(nb, num_pof, pof_size[1], pof_size[0], 1).expand(nb, num_pof, pof_size[1], pof_size[0], 3) *  pof_value.reshape(nb, num_pof, 1, 1, 3)
        pof = pof.permute(0, 1, 4, 2, 3).reshape(nb, num_pof*3, pof_size[1], pof_size[0])
    
        pt_valid_2d = ((pt_valid_2d[:, pof_conns[:, 0]]<3) * (pt_valid_2d[:, pof_conns[:, 1]]<3)).reshape(nb, -1, 1, 1).float()
        pt_valid_3d = (pt_valid_3d[:, pof_conns[:, 0]] * pt_valid_3d[:, pof_conns[:, 1]]).reshape(nb, -1, 1, 1).float()
        if not self.exploit_uvz:
            pof_valid = mask*pt_valid_3d*pt_valid_2d + pt_valid_3d*pt_valid_2d
            pof_valid = pof_valid.unsqueeze(-1).expand(nb, num_pof, pof_size[1], pof_size[0], 3).permute(0,1,4,2,3).reshape(nb, -1, pof_size[1], pof_size[0]).float()
        else:
            pof_valid_a = pt_valid_2d.reshape(nb, num_pof, 1, 1, 1).expand(nb, num_pof, pof_size[1], pof_size[0], 2)
            pof_valid_b = pt_valid_3d.reshape(nb, num_pof, 1, 1, 1).expand(nb, num_pof, pof_size[1], pof_size[0], 1)
            pof_valid = torch.cat([pof_valid_a, pof_valid_b], 4).permute(0, 1, 4, 2, 3).reshape(nb, -1, pof_size[1], pof_size[0])

        return pof, pof_valid

class HM_Generator(object):
    def __init__(self, hm_size=[368, 368], sigma=7, num_kp=18, *args, **kargs): #hm_size <=====> (w, h)
        sigma = max(hm_size[1], hm_size[0]) / 368 * sigma
        self.hm_size, self.sigma, self.num_kp = hm_size, sigma, num_kp
        x_m = np.tile(np.arange(hm_size[0]), (num_kp, hm_size[1], 1))
        y_m = np.tile(np.arange(hm_size[1]), (num_kp, hm_size[0], 1)).transpose([0, 2, 1])
        self.blk_hm = np.stack([x_m, y_m], -1) # num_kp x hm_size[1] x hm_size[0] x 2
        self.scale	= np.array(hm_size).reshape(1, 2)

    def __call__(self, key_points_2d, *args, **kargs): #key_points_2d is num_kp x 3, 3 <====> (x, y, conf)
        pt_xy, pt_valid = (key_points_2d[:, :2]*self.scale).reshape(self.num_kp, 1, 1, 2), key_points_2d[:, 2]
        dist = ((self.blk_hm-pt_xy)**2).sum(-1) # to num_kp x hm_size[1] x hm_size[0]
        hm = np.exp(-dist/(2*self.sigma*self.sigma)) * (pt_valid.reshape(self.num_kp, 1, 1) < 3).astype(np.float)
        hm_valid = (pt_valid < 3).astype(np.float)
        return hm, hm_valid

class HM_Generator_cuda(nn.Module):
    def __init__(self, hm_size=[368, 368], sigma=7, num_kp=21, *args, **kargs):
        super(HM_Generator_cuda, self).__init__()
        sigma = max(hm_size[1], hm_size[0]) / 368 * sigma
        self.hm_size, self.sigma, self.num_kp = hm_size, sigma, num_kp
        x_m = np.tile(np.arange(hm_size[0]), (num_kp, hm_size[1], 1))
        y_m = np.tile(np.arange(hm_size[1]), (num_kp, hm_size[0], 1)).transpose([0, 2, 1])
        blk_hm = np.stack([x_m, y_m], -1).reshape(1, num_kp, hm_size[1], hm_size[0], 2)
        self.register_buffer('blk_hm', torch.from_numpy(blk_hm).float())
        self.register_buffer('scale', torch.tensor(hm_size).float().reshape(1, 1, 2))			

    #inputs batch x self.num_kp x 3
    def forward(self, key_points_2d, *args, **kargs):
        nb = key_points_2d.shape[0]
        pt_xy, pt_valid = (key_points_2d[..., :2]*self.scale).reshape(nb, self.num_kp, 1, 1, 2), key_points_2d[..., 2]
        dist = ((self.blk_hm-pt_xy)**2).sum(-1) # to n x num_kp x hm_size[1] x hm_size[0]
        hm = torch.exp(-dist/(2*self.sigma*self.sigma)) *  (pt_valid.reshape(nb, self.num_kp, 1, 1) < 3) #if not in image, then return a heatmap with pure zeros
        return hm, (pt_valid < 3).float().reshape(nb, self.num_kp)

class HMExtractor(nn.Module):
    def __init__(self, hm_size, num_kp=21):
        super(HMExtractor, self).__init__()
        x_m = np.tile(np.arange(hm_size[0]), (num_kp, hm_size[1], 1))
        y_m = np.tile(np.arange(hm_size[1]), (num_kp, hm_size[0], 1)).transpose([0, 2, 1])
        blk_hm = np.stack([y_m, x_m], -1).reshape(num_kp, hm_size[1], hm_size[0], 2)
        self.register_buffer('blk_hm', torch.from_numpy(blk_hm).unsqueeze(0))

    def forward(self, hms, threshold=0.4, *args, **kargs):
        hb, n_kp, h, w = hms.shape
        conf = (torch.amax(hms.reshape(nb, n_kp, -1), axis=2) > threshold).unsqueeze(-1)
        hms = hms.clamp(0, 1.0)
        f_score = hms.sum(-1).sum(-1).unsqueeze(-1) + 1e-6
        f_pos = hms.unsqueeze(-1) * self.blk_hm
        pos = f_pos.sum(2).sum(2)/f_score
        return torch.cat([pos, conf], 2)

class ToTensor(object):
    def __init__(self, hm_size=[128, 128], pof_size=[128, 128], sigma=7, num_kp=21,dist_thresh=8, hm=True, pof=True, normalize=True, pof_conns=None, *args, **kargs):
        self.hm_gen     = HM_Generator(hm_size=hm_size, sigma=sigma, num_kp=num_kp) if hm else None
        self.pof_gen    = POF_Generator(pof_size=pof_size,pof_conns=pof_conns,dist_thresh=dist_thresh, normalize=normalize) if pof else None
    
    def __call__(self, R):
        dt_name		     =  R['dt_name']
        image		     =  R['image']
        seg_mask         =  R['seg_mask'].astype(np.float)/255.0
        key_points_2d    =  R['key_points_2d'].astype(np.float)
        key_points_3d    =  R['key_points_3d']
        mano_theta	     =  R['mano_theta']
        mano_beta	     =  R['mano_beta']
        w_mask           =  R['w_mask']
        w_ssl            =  R['w_ssl']
        w_pof            =  R['w_pof']
        w_hm             =  R['w_hm']

        w_key_points_2d  =  R['w_key_points_2d']
        w_key_points_3d  =  R['w_key_points_3d']

        h, w, c				 = image.shape
        f					 = np.array([w, h])
        key_points_2d[:,:2] /= f
        if self.hm_gen is None:
            hm, hm_valid         = None, None
        else:
            hm,  hm_valid		 = self.hm_gen(key_points_2d=key_points_2d)
        if self.pof_gen is None:
            pof, pof_valid       = None, None
        else:
            pof, pof_valid		 = self.pof_gen(key_points_2d=key_points_2d, key_points_3d=key_points_3d)

        key_points_2d		 = torch.from_numpy(key_points_2d).float()
        key_points_3d		 = torch.from_numpy(key_points_3d).float()
        image				 = torch.from_numpy(normalize_image(image)).float()
        seg_mask             = torch.from_numpy(seg_mask).float().unsqueeze(0)
        mano_theta			 = torch.from_numpy(mano_theta).float()
        mano_beta			 = torch.from_numpy(mano_beta).float()
        w_mask         		 = torch.tensor(w_mask).float()
        w_ssl                = torch.tensor(w_ssl).float()
        w_pof                = torch.tensor(w_pof).float()
        w_hm                 = torch.tensor(w_hm).float()
        w_key_points_2d      = torch.from_numpy(w_key_points_2d).float()
        w_key_points_3d      = torch.from_numpy(w_key_points_3d).float()

        R = {
            'image'			 : image,
            'seg_mask'       : seg_mask,
            'key_points_2d'  : key_points_2d,
            'key_points_3d'  : key_points_3d,
            'mano_theta'	 : mano_theta,
            'mano_beta'		 : mano_beta,
            'w_mask'         : w_mask,
            'w_ssl'          : w_ssl,
            'w_pof'          : w_pof,
            'w_hm'           : w_hm,
            'w_key_points_2d': w_key_points_2d,
            'w_key_points_3d': w_key_points_3d,
        }

        if hm is not None and hm_valid is not None:
            R['hm']          = torch.from_numpy(hm).float()
            R['hm_valid']    = torch.from_numpy(hm_valid).float()
        if pof is not None and pof_valid is not None:
            R['pof']         = torch.from_numpy(pof).float()
            R['pof_valid']   = torch.from_numpy(pof_valid).float()
        return R

class ProbOp(object):
    def __init__(self, op, p, *args, **kargs):
        self.op, self.p = op, p

    def __call__(self, R):
        return R if random.uniform(0, 1.0) >= self.p else self.op(R)

class Oneof(object):
    def __init__(self, ops, p = None, *args, **kargs):
        self.ops, self.p = ops, p

    def __call__(self, R):
        op = np.random.choice(self.ops, size=1, p=self.p)[0]
        return op(R)

def draw_circle(I, pts, r=1,c=(0,0,255), tag=False, base=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cc = 0
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(I, (x, y), r, c, thickness=-1)
        if tag:
            cv2.putText(I, str(cc+base), (x, y), font, 0.4, (0, 255, 0), 1)
        cc += 1
    return I

def normalize_image(I):
    return (I.astype(np.float)/127.5-1.0).transpose(2,0,1)

def unnormalize_image(tI):
    return ((tI.detach().cpu().numpy() + 1.0)*127.5).transpose(1,2,0).astype(np.uint8)