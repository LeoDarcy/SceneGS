#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    # if cam_info.normal_image is not None and cam_info.normal_image :
    #     resized_normal_image_rgb = PILtoTorch(cam_info.normal_image, resolution)
    #     gt_normal_image = resized_normal_image_rgb[:3, ...]
    # else:
    #     gt_normal_image = cam_info.normal_image
    
    # Modification: Change the resolution
    if cam_info.normal_image is not None and scale > 1:
        # (1, C, H, W)->(C, h, w)
        gt_normal_image = F.interpolate(cam_info.normal_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
        _, H, W = gt_normal_image.shape
        assert H == resolution[1]
        assert W == resolution[0]
        assert gt_image.shape == gt_normal_image.shape, print(gt_normal_image.shape, gt_image.shape)
        # Make sure to work
    else:
        gt_normal_image = cam_info.normal_image
    if cam_info.albedo_image is not None and scale > 1:
        gt_albedo_image = F.interpolate(cam_info.albedo_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
    else:
        gt_albedo_image = cam_info.albedo_image

    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, normal_image=gt_normal_image, albedo_image=gt_albedo_image)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


# Ray helpers from https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
# def get_rays(H, W, K, c2w):
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
#     i = i.t()
#     j = j.t()
#     dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
# Borrow from https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/jupyter/01_casting_rays_that_match_hypersim_images.ipynb

def gen_cam_uv(width_pixels, height_pixels):
    # create grid of uv-values
    u_min  = -1.0
    u_max  = 1.0
    v_min  = -1.0
    v_max  = 1.0
    half_du = 0.5 * (u_max - u_min) / width_pixels
    half_dv = 0.5 * (v_max - v_min) / height_pixels

    u, v = np.meshgrid(np.linspace(u_min+half_du, u_max-half_du, width_pixels),
                    np.linspace(v_min+half_dv, v_max-half_dv, height_pixels)[::-1])

    uvs_2d = np.dstack((u,v,np.ones_like(u)))
    return uvs_2d