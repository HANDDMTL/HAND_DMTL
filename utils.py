
import torch
import torch.nn as nn
from torch import distributed as dist
import torch.nn.functional as F
import numpy as np
from os import path as osp
import math
import cv2
import random
from config import args
cur_dir = osp.dirname(osp.abspath(__file__))

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()
  
def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def reduce_tensor(inp):
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def save_obj(verts, faces, s_path):
    with open(s_path, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in faces:
            if len(f) == 3:
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
            else:
                fp.write( 'f %d %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1, f[3]+1) )

def batch_rodrigues(theta):
    #theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
        2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
        2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_orth_proj(X, camera, keepz=False):
    '''
        X is N x k x 3
        camera is N x 3 (s, tx, ty)
    '''
    x, y, z = X[...,0],X[...,1],X[...,2]
    camera = camera.view(-1, 1, 3)
    s, tx, ty = camera[...,0],camera[...,1],camera[...,2]
    px, py = s*(tx+x), s*(ty+y)
    if keepz==True:
        xy = (torch.stack([px, py], -1) + 1.0) / 2.0
        return torch.cat((xy, (s*z).unsqueeze(-1)),-1)
    else:
        return (torch.stack([px, py],-1) + 1.0) / 2.0

def batch_pers_proj(X, camera, z_cam=200, keepz=False):
    '''
        X is N x num_points x 3
        camera is N x 3 (tx, ty, tz)
    '''
    r_fov = args.fov / 180.0 * math.pi
    t_fov = math.tan(r_fov)
    camera = camera.view(-1, 1, 3)
    x,  y,  z = X[...,0],X[...,1],X[...,2]
    tx, ty, tz = camera[...,0],camera[...,1],camera[...,2]
    x, y, z = tx+x,ty+y,(z_cam+(tz+z)).clamp(1e-6, 100) #avoid inf training error.
    px, py = x/z/t_fov ,y/z/t_fov
    if keepz==True:
        xy = (torch.stack([px,py],-1) + 1.0) / 2.0 # to [0,1)
        return torch.cat((xy, z.unsqueeze(-1)), -1)
    else:
        return (torch.stack([px,py],-1) + 1.0) / 2.0 # to [0,1)

def batch_proj(X, camera, mode='orth', keepz=False):
    if mode == 'orth':
        return batch_orth_proj(X, camera, keepz=keepz)
    else:
        return batch_pers_proj(X, camera, keepz=keepz)

def batch_euler(angles):
    device = angles.device
    batch_size = angles.shape[0]
    angle_x = angles[:,0: 1]
    angle_x_sin, angle_x_cos = torch.sin(angle_x), torch.cos(angle_x)
    angle_y = angles[:,1: 2]
    angle_y_sin, angle_y_cos = torch.sin(angle_y), torch.cos(angle_y)
    angle_z = angles[:,2: 3]
    angle_z_sin, angle_z_cos = torch.sin(angle_z), torch.cos(angle_z)
    batch_zero = torch.tensor([[.0]], dtype=torch.float32).to(device).repeat(batch_size, 1)
    rotation_X = torch.cat((torch.tensor([[1.0, .0, .0, .0]], dtype=torch.float32
            ).to(device).repeat(batch_size, 1),
            angle_x_cos, -angle_x_sin, batch_zero, angle_x_sin, angle_x_cos), dim=1)
    rotation_Y = torch.cat((angle_y_cos, batch_zero, angle_y_sin,
            torch.tensor([[.0, 1.0, .0]], dtype=torch.float32).to(device).repeat(batch_size, 1),
            -angle_y_sin, batch_zero, angle_y_cos), dim=1)
    rotation_Z = torch.cat((angle_z_cos, -angle_z_sin, batch_zero, angle_z_sin, angle_z_cos,
            torch.tensor([[.0, .0, .0, 1.0]], dtype=torch.float32
                    ).to(device).repeat(batch_size, 1)), dim=1)

    rotation_X = rotation_X.view(batch_size, 3, 3)
    rotation_Y = rotation_Y.view(batch_size, 3, 3)
    rotation_Z = rotation_Z.view(batch_size, 3, 3)
    rotation = torch.bmm(torch.bmm(rotation_Z, rotation_Y), rotation_X)
    return rotation


class Agent(nn.Module):
    def __init__(self, mano, model, *args, **kargs):
        super(Agent, self).__init__()
        self.mano  = mano
        self.model = model
    
    def forward(self, I):
        coef3ds, hms, pofs, masks = self.model(I)
        mano_outputs, joints_2ds = [], []

        for coef3d in coef3ds:
            w_shape, w_pose, w_st = torch.split(coef3d, [10, args.n_pca+3, args.n_cam], dim=1)
            w_shape = w_shape.reshape(-1,10)
            w_pose  = w_pose.reshape(-1,args.n_pca+3)
            verts, joints, pose_theta = self.mano(w_pose, w_shape)

            mano_outputs.append({
                'thetas': pose_theta,
                'verts' : verts,
                'betas' : w_shape,
                'joints': joints,
                'w_cam' : w_st,
            })
            
            joints_2ds.append(batch_proj(joints, w_st, args.proj_type))

        return mano_outputs, joints_2ds, hms, pofs, masks

def get_rot_x(radian):
    cv, sv = math.cos(radian), math.sin(radian)
    rot_x = np.zeros([3, 3])
    rot_x[0][0] = 1.0
    rot_x[1][1] = cv
    rot_x[1][2] = -sv
    rot_x[2][1] = sv
    rot_x[2][2] = cv
    return rot_x

def get_rot_y(radian):
    cv, sv = math.cos(radian), math.sin(radian)
    rot_y = np.zeros([3, 3])
    rot_y[1][1] = 1.0
    rot_y[0][0] = cv
    rot_y[0][2] = sv
    rot_y[2][0] = -sv
    rot_y[2][2] = cv
    return rot_y

def get_rot_z(radian):
    cv, sv = math.cos(radian), math.sin(radian)
    rot_z = np.zeros([3, 3])
    rot_z[2][2] = 1.0
    rot_z[0][0] = cv
    rot_z[0][1] = -sv
    rot_z[1][0] = sv
    rot_z[1][1] = cv
    return rot_z

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.initialized = True

    def update(self, val, weight=0.01):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = (1-weight) * self.val + weight * val
        self.avg = self.val

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def reset(self):
        self.initialized = False

def draw_circle(I, pts, r=2,c=(0,0,255)):
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(I, (x, y), r, c, thickness=-1)
    return I

def draw_backbone(I, kp2ds, bones, colors=None):
    if colors is None:
        colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
            [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
            [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
            [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
            [255, 0, 170], [255, 0, 85], [255, 170, 85], [255, 170, 255],
            [85, 85, 255], [85, 255, 85], [255, 85, 85], [255, 170, 170]
        ]
    for (idx, bone) in enumerate(bones):
        x, y = bone
        pa, pb = kp2ds[x, :2], kp2ds[y, :2]
        if len(kp2ds[0]) > 2:
            va, vb = kp2ds[x, 2], kp2ds[y, 2]
            if va < 1 or vb < 1: #not visible
                continue
        x1, y1 = pa
        x2, y2 = pb
        X = (y1, y2)
        Y = (x1, x2)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 2), int(angle), 0, 360, 1)

        cv2.fillConvexPoly(I, polygon, colors[idx])

def extract_key_point2ds(hms, threshold=0.5):
    n_kp, h, w = hms.shape
    hms = hms.reshape(n_kp, -1)
    conf, index = np.amax(hms, axis=1), np.argmax(hms, axis=1)
    y, x = index // w, index % w
    pos = np.stack([x, y]).transpose(1, 0)/np.array([w, h]).reshape(1, -1)
    conf = (conf>threshold)[:, np.newaxis]
    return np.concatenate([pos, conf], axis=1)

def deterministic_training_procedure(seed=2020):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
