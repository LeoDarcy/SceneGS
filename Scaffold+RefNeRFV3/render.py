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
import os
import torch
import cv2
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import Snavely_Tonemap, ACES_Tonemap
from utils.env_utils import cubemap_to_latlong
def MakePanorama(H, W):
    gy, gx = torch.meshgrid(torch.linspace( -0.5 + 1.0 / H, 0.5 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        costheta*sinphi, 
        -sintheta, 
        costheta*cosphi
        ), dim=-1)
    
    return reflvec

def render_env_dr(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_envmap")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    env_mlp = gaussians.mlp_env
    H, W = 256, 512
    lighting = cubemap_to_latlong(gaussians._env_color, (H, W) )#(1, 6, res, res,3)
    # print(uv.shape, infer.shape)
    cv2.imwrite(os.path.join(render_path, "env.hdr"),lighting.cpu().numpy())
    infer = lighting.permute(2, 0, 1) # (3, H, W)
    infer = ACES_Tonemap(infer)
    infer = torch.clamp(infer, 0, 1)
    torchvision.utils.save_image(infer, os.path.join(render_path, "env.png"))
def render_env(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_envmap")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    env_mlp = gaussians.mlp_env
    H, W = 256, 512
    uv = MakePanorama(H, W)
    uv = uv.reshape((-1, 3))
    uv = uv / uv.norm(dim=1, keepdim=True)
    # print(uv.shape)
    infer = env_mlp(uv)
    infer = infer.reshape((H, W, 3))
    # print(uv.shape, infer.shape)
    cv2.imwrite(os.path.join(render_path, "env.hdr"),infer.cpu().numpy())
    infer = infer.permute(2, 0, 1)
    infer = Snavely_Tonemap(infer)
    infer = torch.clamp(infer, 0, 1)
    torchvision.utils.save_image(infer, os.path.join(render_path, "env.png"))
    
    


from einops import rearrange
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, rendering_args={"env_color"})
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # Only for normals
        rendering = render_pkg["normals"]
        gt = view.original_normal_image[0:3, :, :]
        gt = (gt+1)/2.0
        name_list.append('{0:05d}'.format(idx) + ".png")
        # print(rendering)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}_normals'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}_normals'.format(idx) + ".png"))
        torchvision.utils.save_image(torch.cat((rendering, gt), dim=1), os.path.join(gts_path, '{0:05d}_mynormals'.format(idx) + ".png"))

        depths = render_pkg["depth"]
        depths = depths / torch.max(depths)
        torchvision.utils.save_image(depths, os.path.join(render_path, '{0:05d}_depth'.format(idx) + ".png"))
        # 将深度图像转换为彩色图像
        depths = rearrange(depths, "c h w -> h w c")
        depths = depths.cpu().numpy()[:,:,0:1]
        depth_color_image = cv2.applyColorMap((depths*255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(render_path, '{0:05d}_depth_color'.format(idx) + ".png"), depth_color_image)

        alpha = render_pkg["alpha"]
        print("attention ", torch.min(alpha))
        alpha = alpha / torch.max(alpha)
        torchvision.utils.save_image(alpha, os.path.join(render_path, '{0:05d}_alpha'.format(idx) + ".png"))

        tint = render_pkg["tint"]
        print("Tint values ", torch.min(tint), torch.max(tint))
        # alpha = alpha / torch.max(alpha)
        tint = ACES_Tonemap(tint)
        tint = torch.clamp(tint, 0, 1)
        torchvision.utils.save_image(tint, os.path.join(render_path, '{0:05d}_tints'.format(idx) + ".png"))

        roughness = render_pkg["roughness"]
        print("Roughness values ", torch.min(tint), torch.max(tint))
        # alpha = alpha / torch.max(alpha)
        roughness = (roughness +1 )/2
        roughness = torch.clamp(roughness, 0, 1)
        torchvision.utils.save_image(roughness, os.path.join(render_path, '{0:05d}_roughness'.format(idx) + ".png"))

        albedo = torch.clamp(render_pkg["base_color"], 0.0, 1.0)
        gt_albedo = torch.clamp(view.original_albedo_image[0:3, :, :], 0.0, 1.0)
        torchvision.utils.save_image(albedo, os.path.join(render_path, '{0:05d}_albedo'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_albedo, os.path.join(gts_path, '{0:05d}_albedo'.format(idx) + ".png"))

        torchvision.utils.save_image(torch.cat((render_pkg["render"], rendering, tint, roughness), dim=1), os.path.join(gts_path, '{0:05d}_mynormals'.format(idx) + ".png"))
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # print(f"HAHA You load the pretrain checkpoint from ")
        # scene.load(checkpoint)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            render_env_dr(dataset.model_path, "bjytrain", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            render_set(dataset.model_path, "bjytrain", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_env_dr(dataset.model_path, "bjytrain", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            render_set(dataset.model_path, "bjytest", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
