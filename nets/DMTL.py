
import sys
from os import path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from nets.ops import BottleNeck, conv2d, linear_layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TAM(nn.Module):
    def __init__(self, n_task, n_ch):
        super(TAM, self).__init__()
        self.n_task = n_task
        self.convs = nn.Sequential(
            conv2d(c_in=n_ch*self.n_task, c_out=n_ch*self.n_task, k=1, s=1, p=0, g=1, relu=True, bn=True),
            conv2d(c_in=n_ch*self.n_task, c_out=n_ch, k=1, s=1, p=0, g=1, relu=True, bn=True),
            conv2d(c_in=n_ch, c_out=n_ch, k=1, s=1, p=0, g=1, relu=True, bn=True),
            conv2d(c_in=n_ch, c_out=n_ch, k=1, s=1, p=0, g=1, relu=True, bn=True),
            conv2d(c_in=n_ch, c_out=n_ch*self.n_task, k=1, s=1, p=0, g=1, relu=True, bn=True),
            conv2d(c_in=n_ch*self.n_task, c_out=n_ch*self.n_task, k=1, s=1, p=0, g=1, relu=False, bn=False),
        )

    def forward(self, t_xs, *args, **kargs):
        nb, nc = t_xs[0].shape[:2]
        of = torch.stack([F.adaptive_avg_pool2d(t_x, (1, 1)) for t_x in t_xs], -1).reshape(nb, nc*self.n_task, 1, 1)
        atten_mask = F.softmax(self.convs(of).reshape(nb, nc, 1, 1, self.n_task), dim=4).permute(4, 0, 1, 2, 3)
        return sum([atten*t_x for (atten, t_x) in zip(atten_mask, t_xs)])

class Encoder(nn.Module):
    def __init__(self, channel_in=128, channel_per_block=[256, 512, 1024, 2048], layers_per_block=[3,4,6,3], rt_pyramids=True):
        super(Encoder, self).__init__()
        self.ops = nn.ModuleList()
        self.rt_pyramids = rt_pyramids
        for i_blk, (channel, c_layer) in enumerate(zip(channel_per_block, layers_per_block)):
            ops = []
            for i_layer in range(c_layer):
                if i_blk==0 and i_layer==0:
                    c_in  = channel_in
                    c_out = channel
                elif i_layer==0:
                    c_in  = channel_per_block[i_blk-1]
                    c_out = channel
                else:
                    c_in  = channel
                    c_out = channel
                ops.append(BottleNeck(in_plane=c_in, out_plane=c_out, reduce_ratio=4, min_plane=64, residual=True, s=2 if i_layer==0 and i_blk>0 else 1))
            self.ops.append(nn.Sequential(*ops))
    
    def forward(self, I, *args, **kargs):
        x = [I]
        for op in self.ops:
            x.append(op(x[-1]))
        return x[::-1][:-1] if self.rt_pyramids else x[-1]

class Decoder(nn.Module):
    def __init__(self, channel_settings, output_shape=[96, 96], c_out=18):
        super(Decoder, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples = [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = self._predict(output_shape, c_out)

    def _lateral(self, input_size):
        return conv2d(c_in=input_size, c_out=256, k=1,s=1,p=0,g=1,relu=True,bn=True,bias=False)

    def _upsample(self):
        layers = []
        layers.append(conv2d(c_in=256,c_out=256,k=3,s=1,p=1,g=1,relu=True,bn=True,bias=False))
        layers.append(conv2d(c_in=256,c_out=256,k=3,s=1,p=1,g=1,relu=True,bn=True,bias=False))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(conv2d(c_in=256,c_out=256,k=1,s=1,p=0,g=1,relu=False,bn=True))
        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(conv2d(c_in=256,c_out=256,k=3,s=1,p=1,g=1,relu=True,bn=True,bias=False))
        layers.append(conv2d(c_in=256,c_out=256,k=3,s=1,p=1,g=1,relu=True,bn=True,bias=False))
        layers.append(conv2d(c_in=256,c_out=num_class,k=3,s=1,p=1,g=1,relu=False,bn=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
        return feature, self.predict(feature)

class DMTL(nn.Module):
    def __init__(self, stage=4, n_hm=21, n_pof=21, dim_hm=1, dim_pof=3, dim_mask=1,channel_per_block=[128, 512, 1024, 2048], layers_per_block=[3,4,6,3], ft_size=[64, 64], crop_size=256, n_feats=85, *args, **kargs):
        super(DMTL, self).__init__()
        self.stem = nn.Sequential(
            conv2d(c_in=3,c_out=64,k=7,s=2,p=3,g=1,relu=True,bn=True,bias=False),
            conv2d(c_in=64,c_out=128,k=7,s=2,p=3,g=1,relu=True,bn=True,bias=False),
        )

        self.stage      = stage
        self.crop_size  = crop_size

        self.encs       = nn.ModuleList()
        self.pof_decs   = nn.ModuleList()
        self.hm_decs    = nn.ModuleList()
        self.mask_decs  = nn.ModuleList()
        self.regressors = nn.ModuleList()
        self.tams       = nn.ModuleList()

        for i_state in range(stage):
            #encoder
            enc        = Encoder(channel_in=128 if i_state==0 else 256+128, channel_per_block=channel_per_block, layers_per_block=layers_per_block)
            self.encs.append(enc)

            #heat-map decoder
            hm_dec     = Decoder(channel_settings=channel_per_block[::-1], c_out=n_hm*dim_hm, output_shape=ft_size)
            self.hm_decs.append(hm_dec)
            
            #pof decoder
            pof_dec    = Decoder(channel_settings=channel_per_block[::-1], c_out=n_pof*dim_pof, output_shape=ft_size)
            self.pof_decs.append(pof_dec)

            #mask decoder
            mask_dec   = Decoder(channel_settings=channel_per_block[::-1], c_out=dim_mask, output_shape=ft_size)
            self.mask_decs.append(mask_dec)

            chs = 128 + 256

            #regressor head
            if i_state==0:
                regressor = nn.Sequential(
                    Encoder(channel_in=chs, channel_per_block=[256, 512, 1024, 2048], layers_per_block=[3, 3, 3, 3], rt_pyramids=False),
                    nn.AdaptiveAvgPool2d(output_size=1),
                    linear_layers([2048, 512, 512, 512, n_feats]),
                )
            else:
                regressor = self.regressors[-1]
            self.regressors.append(regressor)

            #task attention module (TAM)
            tam = TAM(n_task=3, n_ch=256)
            self.tams.append(tam)

    def forward(self, x, *args, **kargs):
        conv_feats = self.stem(x)
        coef3ds, pofs, hms, masks = [], [], [], []
        for stage in range(self.stage):
            reg_feats = [conv_feats]

            x = conv_feats if stage==0 else torch.cat([conv_feats, ft_tam], dim=1)

            x = self.encs[stage](x)

            ft_hm, hm = self.hm_decs[stage](x)
            reg_feats.append(ft_hm)
            hms.append(hm)

            ft_pof, pof = self.pof_decs[stage](x)
            reg_feats.append(ft_pof)
            pofs.append(pof)

            ft_mask, mask = self.mask_decs[stage](x)
            reg_feats.append(ft_mask)
            masks.append(mask)

            ft_tam = self.tams[stage](reg_feats[1:])
            coef3ds.append(self.regressors[stage](torch.cat([conv_feats, ft_tam], dim=1)))

        return coef3ds, hms, pofs, masks