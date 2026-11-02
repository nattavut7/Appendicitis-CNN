"""Reproducible image-/batch-level CT augmentations for R2 ablations.

Care is required: aggressive elastic deformation, flips, and CutMix can be
anatomically implausible. Use clinical review of generated examples.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
import torch

PROFILES = {
    'none': set(),
    'rotation': {'rotation'},
    'crop': {'crop'},
    'noise': {'noise'},
    'rotation_crop': {'rotation','crop'},
    'rotation_crop_noise': {'rotation','crop','noise'},
    'table2_full': {'rotation','flip','affine','elastic','brightness','contrast','noise','cutmix','mixup'},
}


def components_for_profile(profile):
    if profile in PROFILES:
        return set(PROFILES[profile])
    if profile.startswith('full_without_'):
        part = profile[len('full_without_'):]
        if part not in PROFILES['table2_full']:
            raise ValueError(f'Unknown dropped augmentation: {part}')
        return set(PROFILES['table2_full']) - {part}
    raise ValueError(f'Unknown augmentation profile: {profile}')


class CTTransform:
    def __init__(self, cfg, profile='table2_full'):
        self.cfg = cfg
        self.components = components_for_profile(profile)

    def __call__(self, image):
        c = self.cfg
        x = image.copy()
        h,w = x.shape
        if 'rotation' in self.components:
            angle = np.random.uniform(-float(c['rotation_degrees']),float(c['rotation_degrees']))
            m = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
            x = cv2.warpAffine(x,m,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        if 'flip' in self.components and np.random.random()<float(c['flip_probability']):
            x = cv2.flip(x, 0 if np.random.random()<0.5 else 1)
        if 'affine' in self.components and np.random.random()<float(c['affine_probability']):
            scale=np.random.uniform(*map(float,c['affine_scale']))
            shear=math.tan(math.radians(np.random.uniform(-float(c['affine_shear']),float(c['affine_shear']))))
            m=np.array([[scale, shear, (1-scale)*w/2-shear*h/2],
                        [0,scale,(1-scale)*h/2]],dtype=np.float32)
            x=cv2.warpAffine(x,m,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        if 'elastic' in self.components and np.random.random()<float(c['elastic_probability']):
            alpha=np.random.uniform(*map(float,c['elastic_alpha_px']))
            sigma=np.random.uniform(*map(float,c['elastic_sigma_fraction']))*min(h,w)
            # Gaussian smoothing; scale noise back to the chosen max pixel displacement.
            dx=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigma)
            dy=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigma)
            dx=dx/(np.max(np.abs(dx))+1e-8)*alpha
            dy=dy/(np.max(np.abs(dy))+1e-8)*alpha
            gridx,gridy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
            x=cv2.remap(x,gridx+dx,gridy+dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        if 'crop' in self.components and np.random.random()<float(c['crop_probability']):
            frac=np.random.uniform(*map(float,c['crop_scale']))
            hh=max(2,round(h*frac)); ww=max(2,round(w*frac))
            yy=np.random.randint(0,h-hh+1); xx=np.random.randint(0,w-ww+1)
            x=cv2.resize(x[yy:yy+hh,xx:xx+ww],(w,h),interpolation=cv2.INTER_LINEAR)
        if 'brightness' in self.components and np.random.random()<float(c['brightness_probability']):
            x=x+np.random.uniform(-float(c['brightness_limit']),float(c['brightness_limit']))
        if 'contrast' in self.components and np.random.random()<float(c['contrast_probability']):
            x=(x-0.5)*(1+np.random.uniform(-float(c['contrast_limit']),float(c['contrast_limit'])))+0.5
        if 'noise' in self.components and np.random.random()<float(c['gaussian_noise_probability']):
            x=x+np.random.normal(0,np.random.uniform(*map(float,c['gaussian_noise_std'])),x.shape)
        return np.clip(x,0,1).astype(np.float32)


def batch_mix(images, labels, cfg, profile):
    """Return (images, label_a, label_b, lambda); loss supports soft mixing.

    No CutMix/MixUp on validation/test sets. Batch transforms are independent
    of patient splitting: they only mix training images.
    """
    components = components_for_profile(profile)
    if len(images) < 2:
        return images, labels, labels, 1.0
    if 'cutmix' in components and np.random.random()<float(cfg['cutmix_probability']):
        perm=torch.randperm(len(images),device=images.device)
        lo,hi=map(float,cfg['cutmix_patch_fraction'])
        # Relative linear patch size in each spatial dimension.
        fraction=np.random.uniform(lo,hi)
        h,w=images.shape[-2:]
        hh=max(1,round(fraction*h));ww=max(1,round(fraction*w))
        x0=np.random.randint(0,w-ww+1); y0=np.random.randint(0,h-hh+1)
        mixed=images.clone()
        mixed[:,:,y0:y0+hh,x0:x0+ww]=images[perm,:,y0:y0+hh,x0:x0+ww]
        lam=1-(hh*ww)/(h*w)
        return mixed,labels,labels[perm],lam
    if 'mixup' in components and np.random.random()<float(cfg['mixup_probability']):
        perm=torch.randperm(len(images),device=images.device)
        alpha=float(cfg['mixup_alpha'])
        lam=float(np.random.beta(alpha,alpha))
        return lam*images+(1-lam)*images[perm],labels,labels[perm],lam
    return images,labels,labels,1.0
