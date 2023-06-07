import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np

import kornia.augmentation as A

from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import ImageList


STN_ARCH = Registry("STN_ARCH") 

def build_stn_arch(cfg):
    stn_arch = cfg.MODEL.STN_ARCH
    stn_arch = STN_ARCH.get(stn_arch)(cfg)
    stn_arch.to(torch.device(cfg.MODEL.DEVICE))
    return stn_arch



@STN_ARCH.register()
class FIVE_RANDOM_PERSPECTIVE(nn.Module):

    def __init__(self,cfg):
        super(FIVE_RANDOM_PERSPECTIVE,self).__init__()
        self.affine_params = torch.eye(3).repeat(5,1,1)
        self.nbstn = 5

    def make_random_homography(self, scale_x=1, scale_y=1, t_x=0, t_y=0, shear=0, rot=0, lx=0, ly=0):
        T = np.array(
                    [[1,0,t_x],
                     [0,1,t_y],
                     [0,0,1]]
                    )
        rot = np.pi*rot/180
        R = np.array(
                    [[np.cos(rot),np.sin(rot),0],
                     [-np.sin(rot),np.cos(rot),0],
                     [0,0,1]]
                     )

        S = np.array(
                    [[scale_x,shear,0],
                     [0,scale_y,0],
                     [0,0,1]]
                     )

        L = np.array(
                    [[1,0,0],
                     [0,1,0],
                     [lx,ly,1]]
                     )
    
        H = T@R@S@L
    
        return H
    
    def sample_homography(self,):
        sample = np.random.random(4)
        #fixed ranges 
        sx = sample[0]*1.5+0.5
        sy = sample[1]*1.5+0.5
        lx = sample[2]-0.5
        ly = sample[3]-0.5
        H = self.make_random_homography(scale_x=sx,scale_y=sy,lx=lx,ly=ly)

        return H

    def apply_homography(self, img, H, output_image_size, padding_mode="zeros", eps=1e-8):

        B, C, h, w = img.shape

        h_grid, w_grid = output_image_size
        

        xp_dist, yp_dist = torch.meshgrid(torch.arange(-w_grid//2, w_grid//2 ,device=img.device),\
                                torch.arange(-h_grid//2,h_grid//2, device=img.device))

        homogenous = torch.stack([xp_dist.float(), yp_dist.float(), torch.ones((w_grid, h_grid),\
                                device=img.device)]).reshape(1, 3, -1)
        
        max_dim = max(w_grid, h_grid)
        
        homogenous = (homogenous / torch.tensor([(max_dim-1)/2, (max_dim-1)/2, 1], device=img.device).reshape(1,3,1))
        homogenous = homogenous.repeat(B, 1, 1)
        if H.dim() == 2:
            H = H.repeat(B,1,1)
        map_ind  = H.bmm(homogenous)
        #map_ind = (map_ind[:, :-1, :]/map_ind[:, -1, :].unsqueeze(1)).reshape(B, 2, w_grid, h_grid)
        
        #checking for z component of homogenous larger than zero
        mask = torch.abs(map_ind[:, -1, :]) > eps
        scale = torch.where(mask, 1.0 / (map_ind[:, -1, :] + eps), torch.ones_like(map_ind[:, -1, :]))

        map_ind = (map_ind[:, :-1, :] * scale.unsqueeze(1)).reshape(B, 2, w_grid, h_grid) #/map_ind[:, -1, :]

        if w_grid > h_grid:
            map_ind = (map_ind / torch.tensor([1, (h_grid / w_grid)], device=img.device).reshape(1,2,1,1))
        elif w_grid < h_grid:
            map_ind = (map_ind / torch.tensor([(w_grid / h_grid), 1], device=img.device).reshape(1,2,1,1))
            
        map_ind = (map_ind * torch.tensor([1/(w/w_grid), 1/(h/h_grid)], device=img.device).reshape(1,2,1,1))

        
        grid = map_ind.permute(0,3,2,1)
        
        transformed_image = torch.nn.functional.grid_sample(img, grid, align_corners=True,  padding_mode=padding_mode)

        transformed_image[torch.isnan(transformed_image)] = 0
        
        return transformed_image

    def inverse(self,features):
        all_features = []
        for i in range(len(self.affine_params)):
            H = self.affine_params[i].to(features[i].device)
            tffeat = self.apply_homography(features[i],H,features[i].shape[-2:])
            all_features.append(tffeat)

        
        return all_features

    def forward(self,images: ImageList):
        all_images = []
        for i in range(self.affine_params.shape[0]):
            H = torch.tensor(self.sample_homography()).to(images.tensor.device).float()
            self.affine_params[i] = H
            self.params = H
            images_tf = self.apply_homography(images.tensor,torch.inverse(H),images.tensor.shape[-2:])
            images_tf= ImageList(images_tf,[(images_tf.shape[-2],images_tf.shape[-1])]*images_tf.shape[0])
            all_images.append(images_tf)
        
        return all_images


@STN_ARCH.register()
class FIVE_OPT_PERSPECTIVE(FIVE_RANDOM_PERSPECTIVE):

    def __init__(self,cfg):
        super(FIVE_OPT_PERSPECTIVE,self).__init__(cfg)
        self.nbstn = 5
        # self.affine_params = torch.eye(3).repeat(5,1,1)
        self.affine_params = []
        v = torch.rand(5,4)
        v[:,:2]  = v[:,:2]*1.5+0.5
        v[:,2:]  = v[:,2:] - 0.5
        print(v)
         
        self.random_h_param = nn.Parameter(v)


    def make_random_homography_tensor(self, scale_x=1, scale_y=1, t_x=0, t_y=0, shear=0, rot=0, lx=0, ly=0):
        T = torch.tensor(
                    [[1,0,t_x],
                     [0,1,t_y],
                     [0,0,1]]
                    ).float()
        rot = np.pi*rot/180
        R = torch.tensor(
                    [[np.cos(rot),np.sin(rot),0],
                     [-np.sin(rot),np.cos(rot),0],
                     [0,0,1]]
                     ).float()

        # S = torch.tensor(
        #             [[scale_x,shear,0],
        #              [0,scale_y,0],
        #              [0,0,1]]
        #              ).float()

        S = torch.eye(3)
        S[0,0] = scale_x
        S[0,1] = shear
        S[1,1] = scale_y

        # L = torch.tensor(
        #             [[1,0,0],
        #              [0,1,0],
        #              [lx,ly,1]]
        #              ).float()

        L = torch.eye(3)
        L[2,0] = lx
        L[2,1] = ly

        # import pdb;pdb.set_trace()
        H = T.matmul(R.matmul(S.matmul(L)))
    
        return H

    def fixed_homography(self,sx,sy,lx,ly):   

        H = self.make_random_homography_tensor(scale_x=sx,scale_y=sy,lx=lx,ly=ly)

        return H
    
    def forward(self,images):
        all_images = []
        self.affine_params = []

        if self.training:
            storage = get_event_storage()
            for indi , p in enumerate(self.random_h_param):
                for indj, v in enumerate(p):
                    storage.put_scalar(str(indi)+'_'+str(indj),v, smoothing_hint=False)

        for i in range(self.nbstn):
            sx,sy,lx,ly = self.random_h_param[i]
            H = self.fixed_homography(sx=sx,sy=sy,lx=lx,ly=ly).to(images.tensor.device).float()
            
            # self.affine_params[i] = H
            self.affine_params.append(H)
            self.params = H
            images_tf = self.apply_homography(images.tensor,torch.inverse(H),images.tensor.shape[-2:])
            images_tf= ImageList(images_tf,[(images_tf.shape[-2],images_tf.shape[-1])]*images_tf.shape[0])
            all_images.append(images_tf)
        
        return all_images

