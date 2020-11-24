

import argparse
from os import path as osp

parser = argparse.ArgumentParser(description = 'hand capture baseline')

parser.add_argument('--n_cam',type=int,default=3,help='the number of camera parameters.')

#12.6 45.
parser.add_argument('--fov',type=float,default=12.6,help='the fov angle cover the y axis')
parser.add_argument('--lr',type=float,default=1e-3,help='the learning rate.')
parser.add_argument('--wd',type=float,default=1e-5,help='for the weight its decay weight.')

parser.add_argument('--w_2d', type=float, default=20, help='loss weight')
parser.add_argument('--w_3d_dir', type=float, default=10, help='loss weight')
parser.add_argument('--w_silhouette', type=float, default=1e-2, help='loss weight')

parser.add_argument('--w_pof', type=float, default=100, help='loss weight')
parser.add_argument('--w_hm', type=float, default=200, help='loss weight')
parser.add_argument('--w_mask', type=float, default=5, help='loss weight')

parser.add_argument('--w_self_sup_2d', type=float,default=10, help='loss weight')
parser.add_argument('--w_self_sup_mask', type=float,default=1, help='loss weight')

parser.add_argument('--w_beta', type=float, default=1e-1, help='loss weight')
parser.add_argument('--w_theta', type=float, default=1e-1, help='loss weight')

parser.add_argument('--w_reg_theta', type=float,default=10,help='loss weight')
parser.add_argument('--w_reg_beta', type=float, default=10, help='loss weight')


parser.add_argument('--n_pca', type=int,default=45,help='the number of pca components that used for hand reconstruction.')

parser.add_argument('--proj_type',type=str,default='pers',choices=['orth','pers'],help='projection type.')
parser.add_argument('--data_root',type=str,default=r'/home/data/DMTL',help='the root of datasets.')
parser.add_argument('--workers',type=int,default=8,help='the workers of dataloader')
parser.add_argument('--epoch',type=int,default=33,help='the epoches')
parser.add_argument('--batch_size',type=int,default=24,help='the batch size')
parser.add_argument('--crop_size', type=int,default=256,help='the crop size')
parser.add_argument('--seed', type=int, default=2020, help='the seed for gnerating random variables.')

parser.add_argument('--nccl_port', type=int, default=31415,help='the syns port of NCCL')

parser.add_argument('--cache_prob',type=float,default=0.3, help='the probability of caching the augmented samples.')
parser.add_argument('--cache_cap', type=float,default=0.5, help='the capacity of caching the augmented samples.')
parser.add_argument('--use_cuda', type=int, default=1, help='use cuda to generate the heatmaps and pofs.')

parser.add_argument('--resume',type=int,default=0,help='resume or not.')
parser.add_argument('--check_point',type=str,default='checkpoint/20.pth',help='the checkpoint')

args = parser.parse_args()

dt_folder = {
    'hand_HIU'             : osp.join(args.data_root, 'HIU'),
    'hand_cmu'             : osp.join(args.data_root, 'cmu_hand'),
    'hand_FreiHAND'        : osp.join(args.data_root, 'FreiHAND_pub_v2'),
    'hand_RHD'             : osp.join(args.data_root, 'RHD_published_v2'),
    'hand_STB'             : osp.join(args.data_root, 'STB'),
    'hand_dexter'          : osp.join(args.data_root, 'dexter_object'),
}

k_fold = {
    'hand_HIU'             : 1.0/3,
    'hand_cmu'             : 1,
    'hand_FreiHAND'        : 1,
    'hand_RHD'             : 1,
    'hand_STB'             : 1,
    'hand_dexter'          : 1,
}

train_set = ['hand_FreiHAND']

eval_set = ['hand_FreiHAND']

cfg={
    'stage':4, 
    'sigma':16,
    'ch_out':21, 
    'numPOF':21,
    'pof_size':[128, 128],
    'hm_size':[128,128],
    'ft_size':[args.crop_size//4, args.crop_size//4], #w x h
}
