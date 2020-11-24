import os
import torch
import torch.nn as nn

from PIL import Image
import numpy as np

import torch.nn.functional as F

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams
)

class MaskRenderer(nn.Module):
    def __init__(self, device, faces, img_size = 256, batch_size=32, n_verts=778):
        super(MaskRenderer, self).__init__()
        self.img_size = img_size
        self.R, self.T = look_at_view_transform(eye=((0,0,-8),), at=((0, 0, 0),), up=((0, -1, 0),), device = device) 
        cameras = FoVOrthographicCameras(device=device, R=self.R, T=self.T, znear = -100, zfar = 100)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        mask_raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=0,
            faces_per_pixel=100, 
        )

        self.mask_render = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=mask_raster_settings
            ),
            shader=SoftSilhouetteShader()
        )

        self.register_buffer('faces',  faces.repeat((batch_size, 1, 1)))
        self.register_buffer('colors', torch.ones([batch_size, n_verts, 3]))
        self.texture = TexturesVertex(verts_features = self.colors)

    def forward(self, verts, scale2opengl = True):
        if scale2opengl:
            verts = verts*2-1 
        batch_size = verts.shape[0]
        faces, colors = self.faces[:batch_size], self.colors[:batch_size]
        texture = TexturesVertex(verts_features = colors)
        mesh = Meshes(verts = verts, faces = faces, textures = texture)
        masks = self.mask_render(mesh, R=self.R, T=self.T)
        return F.sigmoid(masks[...,3]*6.0)*2.0-1.0

class HandRenderer:
    def __init__(self, device, faces, img_size = 256):
        super(HandRenderer, self).__init__()
        self.img_size = img_size
        self.R, self.T = look_at_view_transform(eye=((0,0,-8),), at=((0, 0, 0),), up=((0, -1, 0),), device = device) 
        cameras = FoVOrthographicCameras(device=device, R=self.R, T=self.T, znear = -100, zfar = 100)
        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            cull_backfaces = True
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -8.0]])
        self.mask_render = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            ),
        )
        self.faces = faces

    def __call__(self, verts, colors = None, scale2opengl = True):
        if scale2opengl:
            verts = verts*2-1   # scale to [-1,1]
        batch_size = verts.shape[0]
        faces = self.faces.repeat((batch_size, 1, 1))
        if colors is None:
            colors = torch.ones_like(verts)*255
        texture = TexturesVertex(verts_features = colors)
        mesh = Meshes(verts = verts, faces = faces, textures = texture)
        imgs = self.mask_render(mesh, R=self.R, T=self.T)[...,3] * 255.0
        return imgs