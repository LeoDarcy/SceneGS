import torch
import numpy as np
import os
from typing import List
from einops import repeat, rearrange
import nvdiffrast.torch as dr
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import Snavely_Tonemap, ACES_Tonemap, getProjectionMatrix
from torch import nn
'''
Generate six fov=90 cubemap rotation for cubemap
From GSIR baking.py
'''
def GenerateCubemapRotation():
    rotations: List[torch.Tensor] = [
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.0, 1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
    ]
    return rotations

def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
class FakeCamera(nn.Module):
    '''
    Fake camera for ray tracing
    '''
    def __init__(self, R: torch.Tensor, T: torch.Tensor, H, W, FoVx, FoVy
                 ):
        super(FakeCamera, self).__init__()
        self.znear = 0.01
        self.zfar = 100.0
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = H
        self.image_width = W
        self.world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def customed_generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        assert False
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    # print(pc._anchor_feat.shape)
    # assert False, visible_mask.shape
    # feat = pc._anchor_feat[visible_mask]
    feat = pc.get_anchor_feature[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    # print("Line 104, Check anchors", visible_mask.shape, visible_mask.sum(), pc.get_anchor.shape, anchor.shape)
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    
    # get normals and tints to calculate RGB
    if pc.add_cov_dist:
        feature1 = torch.cat([feat, ob_view, ob_dist], dim=1)
        normals = pc.get_normals_mlp(feature1)
        tints = pc.get_tint_mlp(feature1)
        roughness = pc.get_roughness_mlp(feature1)
        mentalic = pc.get_mentalic_mlp(feature1)
    else:
        feature2 = torch.cat([feat, ob_view], dim=1)
        normals = pc.get_normals_mlp(feature2)
        tints = pc.get_tint_mlp(feature2)
        roughness = pc.get_roughness_mlp(feature2)
        mentalic = pc.get_mentalic_mlp(feature2)
    normals = normals.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]
    tints = tints.reshape([anchor.shape[0]*pc.n_offsets, 3]) 
    roughness = roughness.reshape([anchor.shape[0]*pc.n_offsets, 3]) 
    mentalic = mentalic.reshape([anchor.shape[0]*pc.n_offsets, 3])
    # get tint to calculate RGB

    

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

   
    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # First get xyz not masked
    scale_repeat = repeat(grid_scaling, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    anchor_repeat = repeat(anchor, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    not_masked_offsets = offsets * scale_repeat[:,:3]
    not_masked_xyz = anchor_repeat + not_masked_offsets
    # Normal has obtained 
    normals_norm = normals/normals.norm(dim=1, keepdim=True) # (N, 3)
    # First get view for neural gaussian
    neural_ob_view = not_masked_xyz - viewpoint_camera.camera_center
    # dist
    neural_ob_dist = neural_ob_view.norm(dim=1, keepdim=True)
    # view
    neural_ob_view = neural_ob_view / neural_ob_dist
    out_view = 2 * normals_norm * torch.sum(normals_norm * neural_ob_view, dim=1, keepdim=True) - neural_ob_view
    out_view = out_view/out_view.norm(dim=1, keepdim=True)
    # env_feat = pc.get_env_mlp(out_view) # [n*10, feature]
    env_color= pc.get_env_mlp(out_view) # [n*10, 3]
    
    # This time we set roughness to a fix value:
    # roughness = 1.0
    # print(out_view.shape, feat.shape, env_feat.shape)
    # feat = feat + env_feat * roughness
    # update cat_feature to get offset's color
    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    base_color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    # print(color.shape, env_color.shape)
    # assert 1==0
    color = base_color * tints #* env_color
    

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, normals,
                                   tints, roughness, mentalic, base_color, env_color], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets, normals, tints, roughness, mentalic, base_color, env_color = masked.split([6, 3, 3, 7, 3, 3, 3, 3, 3, 3, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    
    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, tints, roughness, mentalic, base_color, env_color
    else:
        return xyz, color, opacity, scaling, rot, normals, tints, roughness, mentalic, base_color, env_color

# Modification of render in gaussian_renderer __init__.py
def CustomedRender(viewpoint_camera, pc : GaussianModel, pipe, bg_color, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    viewpoint_camera is a FakeCamera 
    pipe
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, tints, roughness, mentalic, base_color, env_color = customed_generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot, normals, tints, roughness, mentalic, base_color, env_color = customed_generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # Rasterize normal and depth
    # Prepare two type of depth: derive from the depth map and from the alpha blending.
    # Borrorw code from Gaussian Shader Line 179
    # Calculate Gaussians projected depth and predicted normals
    p_hom = torch.cat([xyz, torch.ones_like(xyz[...,:1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
    p_view = p_view[...,:3,:]
    depth = p_view.squeeze()[...,2:3]
    depth = depth.repeat(1,3)

    # get distance 
    distance = torch.sqrt(torch.sum(p_view.squeeze()**2, dim=1))
    

    render_extras = { "normals": normals, "base_color": base_color, 
                     "tint": tints}#"depth": depth,"env_color": env_color, , "roughness": roughness, "distance": distance, "mentalic": mentalic

    #Get normals (shortest aixs)
    # normal_axis = pc.get_minimum_axis
    
    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        # print("Now is ", k, render_extras[k].shape)
        image = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)[0]
        # print("Out put", image.shape)
        if k == "normals":
            out_extras[k] = torch.nn.functional.normalize(image, p=2, dim=0)
        else:
            out_extras[k] = image

    # Rasterize visible Gaussians to alpha mask image. 
    raster_settings_alpha = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(xyz) 
    out_extras["alpha"] =  rasterizer_alpha(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)[0]
    torch.cuda.empty_cache()
    
    # rendered_image = (tint) *(albedo)
    
    
    
    # Ray tracing basd on ray_o and ray_d
    
    
    # final_color = base_color * tint + roughness * light_color * light_tint


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        out = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        out = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    out.update(out_extras)
    return out



def get_neural_gaussians(viewpoint_camera, pc : GaussianModel, pipe, anchor, rays_o, rays_d, is_training=False):
    # anchors [N,3]  
    feat = pc.mlp_anchor_feature(anchor)

    ## get view properties for anchor
    ob_view = anchor - rays_o
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = rays_d

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    
    # get normals and tints to calculate RGB
    if pc.add_cov_dist:
        feature1 = torch.cat([feat, ob_view, ob_dist], dim=1)
        normals = pc.get_normals_mlp(feature1)
        tints = pc.get_tint_mlp(feature1)
        roughness = pc.get_roughness_mlp(feature1)
        mentalic = pc.get_mentalic_mlp(feature1)
    else:
        feature2 = torch.cat([feat, ob_view], dim=1)
        normals = pc.get_normals_mlp(feature2)
        tints = pc.get_tint_mlp(feature2)
        roughness = pc.get_roughness_mlp(feature2)
        mentalic = pc.get_mentalic_mlp(feature2)
    normals = normals.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]
    tints = tints.reshape([anchor.shape[0]*pc.n_offsets, 3]) 
    roughness = roughness.reshape([anchor.shape[0]*pc.n_offsets, 3]) 
    mentalic = mentalic.reshape([anchor.shape[0]*pc.n_offsets, 3])
    # get tint to calculate RGB

    

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

   
    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    
    
    # offsets
    # offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # # First get xyz not masked
    # scale_repeat = repeat(grid_scaling, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # anchor_repeat = repeat(anchor, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # not_masked_offsets = offsets * scale_repeat[:,:3]
    # not_masked_xyz = anchor_repeat + not_masked_offsets
    # # Normal has obtained 
    # normals_norm = normals/normals.norm(dim=1, keepdim=True) # (N, 3)
    # # First get view for neural gaussian
    # neural_ob_view = not_masked_xyz - viewpoint_camera.camera_center
    # # dist
    # neural_ob_dist = neural_ob_view.norm(dim=1, keepdim=True)
    # # view
    # neural_ob_view = neural_ob_view / neural_ob_dist
    # out_view = 2 * normals_norm * torch.sum(normals_norm * neural_ob_view, dim=1, keepdim=True) - neural_ob_view
    # out_view = out_view/out_view.norm(dim=1, keepdim=True)
    # # env_feat = pc.get_env_mlp(out_view) # [n*10, feature]
    # env_color= pc.get_env_mlp(out_view) # [n*10, 3]
    
    # This time we set roughness to a fix value:
    # roughness = 1.0
    # print(out_view.shape, feat.shape, env_feat.shape)
    # feat = feat + env_feat * roughness
    # update cat_feature to get offset's color
    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    base_color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    # print(color.shape, env_color.shape)
    # assert 1==0
    color = (base_color) * tints #* env_color
    
    color = rearrange(color, '(n o) c -> n o c', o=pc.n_offsets, c=3)
    color = torch.mean(color, dim=1)
    return color

    # combine for parallel masking
    # concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    # concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, normals, tints, roughness, mentalic, base_color, env_color], dim=-1)
    # masked = concatenated_all[mask]
    # scaling_repeat, repeat_anchor, color, scale_rot, offsets, normals, tints, roughness, mentalic, base_color, env_color = masked.split([6, 3, 3, 7, 3, 3, 3, 3, 3, 3, 3], dim=-1)
    
    # # post-process cov
    # scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    
    # rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # # post-process offsets to get centers for gaussians
    # offsets = offsets * scaling_repeat[:,:3]
    # xyz = repeat_anchor + offsets

    
    # if is_training:
    #     return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, tints, roughness, mentalic, base_color, env_color
    # else:
    #     return xyz, color, opacity, scaling, rot, normals, tints, roughness, mentalic, base_color, env_color