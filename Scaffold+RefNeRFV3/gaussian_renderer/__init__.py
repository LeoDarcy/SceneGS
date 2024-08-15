#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inKria.fr
#
import torch
import random
from einops import repeat, rearrange
import nvdiffrast.torch as dr
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import Snavely_Tonemap, ACES_Tonemap, getProjectionMatrix, normal_from_depth_image
from gaussian_renderer.baking_utils import GenerateCubemapRotation, FakeCamera, CustomedRender, get_neural_gaussians
from utils.camera_utils import gen_cam_uv
import scene.ref_utils as ref_utils
def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref
def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    # feat = pc.get_anchor_feature[visible_mask]

    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

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
    # if pc.add_cov_dist:
    #     feature1 = torch.cat([feat, ob_view, ob_dist], dim=1)
    #     normals = pc.get_normals_mlp(feature1)
    #     tints = pc.get_tint_mlp(feature1)
    #     # roughness = pc.get_roughness_mlp(feature1)
    #     roughness = pc.get_roughness(feature1) # Softplus for activation in RefNeRF
    #     mentalic = pc.get_mentalic_mlp(feature1)
    # else:
    #     feature2 = torch.cat([feat, ob_view], dim=1)
    #     normals = pc.get_normals_mlp(feature2)
    #     tints = pc.get_tint_mlp(feature2)
    #     # roughness = pc.get_roughness_mlp(feature2)
    #     roughness = pc.get_roughness(feature2) # Softplus for activation in RefNeRF
    #     mentalic = pc.get_mentalic_mlp(feature2)
    normals = pc.get_normals_mlp(feat)
    tints = pc.get_tint_mlp(feat)
    roughness = pc.get_roughness(feat)
    mentalic = pc.get_mentalic_mlp(feat)
    
    normals = normals.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]
    tints = tints.reshape([anchor.shape[0]*pc.n_offsets, 3]) 
    # print("check ", roughness)
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
    normals = normals/normals.norm(dim=1, keepdim=True) # (N, 3)
    # First get view for neural gaussian
    neural_ob_view = not_masked_xyz - viewpoint_camera.camera_center
    # dist
    neural_ob_dist = neural_ob_view.norm(dim=1, keepdim=True)
    # view
    neural_ob_view = neural_ob_view / neural_ob_dist
    out_view = 2 * normals_norm * torch.sum(normals_norm * neural_ob_view, dim=1, keepdim=True) - neural_ob_view
    out_view = out_view/out_view.norm(dim=1, keepdim=True)
    # env_feat = pc.get_env_mlp(out_view) # [n*10, feature]
    # env_color= pc.get_env_mlp(out_view) # [n*10, 3]
    #Fake env_color
    env_color = tints.clone()
    
    # This time we set roughness to a fix value:
    # roughness = 1.0
    # print(out_view.shape, feat.shape, env_feat.shape)
    # feat = feat + env_feat * roughness
    # update cat_feature to get offset's color
    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    # if pc.appearance_dim > 0:
    #     if pc.add_color_dist:
    #         color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
    #     else:
    #         color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    # else:
    #     if pc.add_color_dist:
    #         color = pc.get_color_mlp(cat_local_view)
    #     else:
    #         color = pc.get_color_mlp(cat_local_view_wodist)
    color = pc.get_color_mlp(feat)
    base_color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    # print(color.shape, env_color.shape)
    # assert 1==0
    color = base_color# * tints #* env_color
    

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, normals, tints, roughness, mentalic, base_color, env_color], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets, normals, tints, roughness, mentalic, base_color, env_color = masked.split([6, 3, 3, 7, 3, 3, 3, 3, 3, 3, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    my_feat = repeat(feat, 'n (c) -> (n k) (c)', k=pc.n_offsets)[mask]
    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, tints, roughness, mentalic, base_color, env_color, my_feat
    else:
        return xyz, color, opacity, scaling, rot, normals, tints, roughness, mentalic, base_color, env_color, my_feat

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, rendering_args=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, tints, roughness, mentalic, base_color, env_color, my_feat = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot, normals, tints, roughness, mentalic, base_color, env_color, my_feat = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass
    if True:#RefNeRF
        # Get specular color:
        neural_ob_view = xyz - viewpoint_camera.camera_center
        # dist
        neural_ob_dist = neural_ob_view.norm(dim=1, keepdim=True)
        # view
        neural_ob_view = neural_ob_view / neural_ob_dist
        out_view = 2 * normals * torch.sum(normals * neural_ob_view, dim=1, keepdim=True) - neural_ob_view
        out_view = out_view/out_view.norm(dim=1, keepdim=True)
        # refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
        
        dir_enc = pc.dir_enc_fn(out_view, roughness[:, 0:1])#(N, 72)
        # dir_enc = dir_enc[0]

        dotprod = torch.sum(normals * neural_ob_view, dim=1, keepdim=True)
        # print("check ", my_feat.shape, dir_enc.shape, dotprod.shape,)
        x = torch.cat([my_feat, dir_enc, dotprod], dim=-1)
        # print("check ", my_feat.shape, dir_enc.shape, dotprod.shape, x.shape)
        x = x.reshape(-1, x.shape[-1])
        speular_color = pc.get_ref_specular(x)
        env_color = speular_color
        color =  torch.clamp(
            ref_utils.linear_to_srgb(speular_color * tints + color), 0.0, 1.0
        )
    if False:
        pc.brdf_mlp.build_mips()
        gb_pos = xyz # (N, 3) 
        view_pos = viewpoint_camera.camera_center.repeat(opacity.shape[0], 1) # (N, 3) 

        diffuse = base_color # (N, 3) 
        normal = normals
        specular = mentalic# tints # (N, 3) 
        roughness_input = roughness[:,0:1] # (N, 1) 
        # assert gb_pos.shape[0] > 0
        # assert gb_pos.shape == specular.shape
        # print("Check ", gb_pos.shape, normal.shape, diffuse.shape, specular.shape, roughness.shape, view_pos.shape)
        color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness_input[None, None, ...], view_pos[None, None, ...])

        color = color.squeeze() # (N, 3) 
        diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
        specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 
        

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
    
    
    # distance = torch.sqrt(torch.sum((p_view-viewpoint_camera.camera_center)**2, dim=1))
    # distance = distance.unsqueeze(-1).repeat(1, 3)
    distance = p_view.squeeze() - viewpoint_camera.camera_center

    altered_normal = normals * 0.5 + 0.5# range (-1, 1) -> (0, 1)
    render_extras = {"depth": depth, "normals": altered_normal, "base_color": base_color, "env_color": env_color,
                     "tint": tints, "roughness": roughness, "distance": distance , "mentalic": mentalic}
    #Not use to save disk 
    #Get normals (shortest aixs)
    # normal_axis = pc.get_minimum_axis
    
    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        image = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)[0]
        if k == "normals":
            # out_extras[k] = torch.nn.functional.normalize(image, p=2, dim=0)
            image = (image - 0.5) * 2.
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
    # Render normal from depth image, and alpha blend with the background. 
    out_extras["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], bg_color=bg_color, alpha=out_extras['alpha'][0])
    # Tonemap to (0,1)
    # rendered_image = Snavely_Tonemap(rendered_image)
    # rendered_image = ACES_Tonemap(rendered_image)
    # rendered_image = (Env + tint) *(albedo) + (Env + tint) * reflection
    
    if rendering_args is not None:
        # Get view_direction
        normal_images, depth_images = out_extras["normals"],  out_extras["depth"] #(3, H, W)
        
        # Use GT
        if "gt_normal" in rendering_args:
            print("Attention: we use gt") 
            normal_images = viewpoint_camera.original_normal_image.cuda() #(c, h, w)
        normal_images = rearrange(normal_images, "c h w -> (h w) c")
        # detach 
        normal_images = normal_images.float().detach()
        # distance_images = rearrange(distance_images, "c h w -> (h w) c")
        depth_images = rearrange(depth_images, "c h w -> (h w) c")
        # assert distance_images.isnan().any() == False, distance_images.isnan().any()

        uv = gen_cam_uv(viewpoint_camera.image_width, viewpoint_camera.image_height).reshape(-1, 3)
        uv = torch.tensor(uv).float().to(normal_images.device)
        # uv = torch.cat([uv, torch.ones_like(uv[...,:1])], -1)
        # print("Line 309", uv.shape)
        
        pt3d = viewpoint_camera.projection_matrix[:3, :3] @ uv.T #(3, N)

        pt3d = pt3d / torch.norm(pt3d, dim=0, keepdim=True) #(3, N)

        origin_pt3d = pt3d.clone()# * 1.0

        pt3d = pt3d * (depth_images.T)
        c2w = viewpoint_camera.c2w_transform
        pt3d = c2w[:3, :3] @ pt3d #(3, N)

        # Origin pointout in world coordinates
        origin_pt3d = c2w[:3, :3] @ origin_pt3d #(3, N)
        origin_pointout = (origin_pt3d / torch.norm(origin_pt3d, dim=0, keepdim=True)).T
        assert origin_pointout.isnan().any() == False, origin_pointout.isnan().any()

        #有可能depth=0
        # print("PTS 3d" , torch.min(torch.sum(pt3d, dim=0)), torch.min(depth_images))
        assert pt3d.isnan().any() == False, pt3d.isnan().any()
        rays_o = pt3d.T + viewpoint_camera.camera_center #(N, 3)
        rays_o_withzeros = pt3d.T + viewpoint_camera.camera_center #(N, 3)
        # print("rays_o 3d" , torch.min(torch.sum(rays_o, dim=1)))
        assert rays_o.isnan().any() == False, rays_o.isnan().any()
        assert torch.norm((rays_o - viewpoint_camera.camera_center), dim=1, keepdim=True).isnan().any() == False, "out view norm has nan"
        
        # norm_value = torch.sum((rays_o - viewpoint_camera.camera_center), dim=1, keepdim=True)
        # print("norm value", torch.min(torch.abs(norm_value)), norm_value.shape)
        
        
        out_views = (rays_o - viewpoint_camera.camera_center) / torch.norm((rays_o - viewpoint_camera.camera_center), dim=1, keepdim=True)
        

        rays_d_with_nan = 2 * normal_images * torch.sum(normal_images * out_views, dim=1, keepdim=True) - out_views
        nan_mask = rays_d_with_nan.isnan()

        rays_d = rays_d_with_nan[nan_mask == False].reshape((-1, 3))
        rays_o = rays_o[nan_mask == False].reshape((-1, 3))
        # print("Check shape, ", rays_o.shape, rays_d.shape, out_views.shape, nan_mask.shape)
        assert rays_d.isnan().any() == False, rays_d.isnan().any()
        rays_d = rays_d/rays_d.norm(dim=1, keepdim=True) #[N, 3]
        
        assert rays_d.isnan().any() == False, rays_d.isnan().any()
        assert rays_o.isnan().any() == False, rays_o.isnan().any()

        # makes env rays_d
        rays_d_with_nan[nan_mask == True] = origin_pointout[nan_mask == True] #replace with ray out in world coordinates
        
        assert rays_d_with_nan.isnan().any() == False, rays_d_with_nan.isnan().any()
        
        # Return tint* base_color:
        # light = Baking(viewpoint_camera, pc, pipe, rays_o, rays_d, normal_images, bg_color, retain_grad=retain_grad) # 给定rayo, 可以见的到 （H,W, 1）个Cubmap表示
        roughness_image = out_extras["roughness"] #(C, H, W)
        if rendering_args is not None and "env_color" in rendering_args:
            # env_light = Bounce_env(viewpoint_camera, pc, pipe, rays_o, rays_d_with_nan, normal_images, bg_color, retain_grad=retain_grad) # 查对应的env
            diffuse_map = rearrange(rendered_image, "c h w -> (h w) c").reshape((-1, 3))
            specular_map = rearrange(roughness_image, "c h w -> (h w) c").reshape((-1, 3))
            roughness_map = rearrange(roughness_image, "c h w -> (h w) c").reshape((-1, 3))
            # print(rays_o_withzeros.shape, rays_d_with_nan.shape, normal_images.shape, diffuse_map.shape, specular_map.shape, roughness_map.shape)
            env_light, _ = Shade_Env(viewpoint_camera, pc, rays_o_withzeros, rays_d_with_nan, normal_images, 
                                diffuse_map, specular_map, roughness_map) 
            assert env_light.isnan().any() == False, env_light.isnan().any()
            # print("ENV Envole!", env_light.shape)
            # rendered_image = rendered_image + roughness_image * env_light 
            rendered_image = env_light# rendered_image + roughness_image * env_light 
        if rendering_args is not None and "bounce" in rendering_args:
            
            light = Bounce(viewpoint_camera, pc, pipe, rays_o, rays_d, normal_images, bg_color, retain_grad=retain_grad) # 直接查找对应的位置
            # Add environment map
            nan_mask = rearrange(nan_mask, "(h w) c -> c h w", h=rendered_image.shape[1], w=rendered_image.shape[2] )
            rendered_image[nan_mask==False] = rendered_image[nan_mask==False] + roughness_image[nan_mask==False] * light.reshape((-1)) # Add reflection light 

    assert rendered_image.isnan().any() == False, rendered_image.isnan().any()
    # rendered_image = ACES_Tonemap(rendered_image)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    
    # Render normal from depth image, and alpha blend with the background. 
    # out_extras["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], bg_color=bg_color, alpha=out_extras['alpha'][0])

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

def Shade_Env(viewpoint_camera, pc : GaussianModel, rays_o, rays_d, normals_map,  diffuse_map, tint_map, roughness_map):
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    # 实际上每个位置进行baking出对应的位置
    N_pixels = rays_d.shape[0]
    assert N_pixels == H * W
    pc.brdf_mlp.build_mips()
    gb_pos = rays_o #(N, 3)
    normal = normals_map
    diffuse = diffuse_map
    specular = tint_map
    roughness = roughness_map[:,0:1]
    view_pos = viewpoint_camera.camera_center.repeat(N_pixels, 1) # (N, 3) 
    color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], 
                                        diffuse[None, None, ...], specular[None, None, ...], 
                                        roughness[None, None, ...], view_pos[None, None, ...])
    out_rgb = color.squeeze() # (N, 3)
    out_rgb = rearrange(out_rgb, "(h w) c -> c h w", h=H, w=W)
    diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
    diffuse_color = rearrange(diffuse_color, "(h w) c -> c h w", h=H, w=W)
    brdf_pkg['diffuse_img'] = diffuse_color
    specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 
    specular_color = rearrange(specular_color, "(h w) c -> c h w", h=H, w=W)
    brdf_pkg['specular_img'] = specular_color
    return out_rgb, brdf_pkg

'''
Get the Gaussians based on the ray_o, ray_d, normals_map
Note that these items can not share the same camera paramters to rasterization
'''
def find_nearest_projection(anchor, ro, rd):
    # Make sure rd is normalized

    anchor_ex = anchor.unsqueeze(0).expand( rd.shape[0], -1,-1) #(N, 3)->(K，N,  3)
    ro_ex = ro.unsqueeze(1).expand(-1, anchor.shape[0], -1)#(K,3)->(K, N, 3)
    rd_ex = rd.unsqueeze(1).expand(-1, anchor.shape[0], -1)#(K,3)->(K, N, 3)
    anchor_dir = anchor_ex - ro_ex
    
    #先基于点乘把正向的anchor筛选出来
    cos_anchor_ray_d = torch.sum(anchor_dir * rd_ex, dim=2, keepdim=False) >= 0 #(K,N)
    cos_anchor_ray_d = torch.sum(cos_anchor_ray_d, dim=0) > 0 #(N)
    anchor = anchor[cos_anchor_ray_d] #(N, 3)
    assert anchor.isnan().any() == False, anchor.isnan().any()
    del ro_ex
    del rd_ex
    del anchor_dir
    del cos_anchor_ray_d
    torch.cuda.empty_cache()
    #第二步，重新算一下叉乘，来算投影距离
    anchor_ex = anchor.unsqueeze(0).expand( rd.shape[0], -1,-1) #(N, 3)->(K，N,  3)
    ro_ex = ro.unsqueeze(1).expand(-1, anchor.shape[0], -1)#(K,3)->(K, N, 3)
    rd_ex = rd.unsqueeze(1).expand(-1, anchor.shape[0], -1)#(K,3)->(K, N, 3)
    anchor_dir = anchor_ex - ro_ex
    assert anchor_dir.isnan().any() == False, anchor_dir.isnan().any()
    assert rd_ex.isnan().any() == False, rd_ex.isnan().any()

    distance = torch.cross(anchor_dir, rd_ex, dim=2) #(K, N, 3)
    assert distance.isnan().any() == False, distance.isnan().any()
    distance = torch.norm(distance, dim=2)#K, N,
    assert distance.isnan().any() == False, distance.isnan().any()
    _, indices = torch.min(distance, dim=1)
    assert indices.isnan().any() == False, indices.isnan().any()
    selected_anchor = anchor[indices] #(K,3)
    assert selected_anchor.isnan().any() == False, selected_anchor.isnan().any()
    dis = torch.sum((selected_anchor-ro) * rd, dim=1, keepdim=True)
    points = ro + dis * rd

    del ro_ex
    del rd_ex
    del distance
    del anchor_dir
    torch.cuda.empty_cache()
    return points
def find_nearest_points(A, B):
    # A is target, B is point in pixels
    dists = torch.norm(A[:, None] - B, dim=2)
    min_dists, indices = torch.min(dists, dim=0)
    result = A[indices]
    return result
def Bounce(viewpoint_camera, pc : GaussianModel, pipe, rays_o, rays_d, normals_map, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    else:
        assert False # The visible mask must be re-calculated!
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    # 实际上每个位置进行baking出对应的位置
    N_pixels = rays_d.shape[0]
    out_rgb = torch.zeros_like(rays_d)
    out_tint = torch.zeros_like(rays_d)
    out_depth = torch.zeros_like(rays_d)

    projected_points = find_nearest_projection(pc.get_anchor, rays_o, rays_d)
    assert projected_points.isnan().any() == False, projected_points.isnan().any()
    reflected_colors = get_neural_gaussians(viewpoint_camera, pc, pipe, projected_points, rays_o, rays_d,is_training=retain_grad)
    assert reflected_colors.isnan().any() == False, reflected_colors.isnan().any()
    # reflected_colors = rearrange(reflected_colors, '(h w) c -> c h w', h=int(viewpoint_camera.image_height), w=int(viewpoint_camera.image_width))
    return reflected_colors
    

def Bounce_env(viewpoint_camera, pc : GaussianModel, pipe, rays_o, rays_d, normals_map, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    else:
        assert False # The visible mask must be re-calculated!
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    # 实际上每个位置进行baking出对应的位置
    N_pixels = rays_d.shape[0]
    assert N_pixels == H * W
    out_rgb = torch.zeros_like(rays_d)
    env_color = pc._env_color #(1, 6, res, res,3)
    rds = rearrange(rays_d, '(h w) c-> 1 h w c', h=H, w=W)
    rgb_envmap = dr.texture(env_color,  # [1, 6, 16, 16, 3]
                            rds.contiguous(),  # [1, H, W, 3]
                            filter_mode="linear",
                            boundary_mode="cube",)
    rgb = rgb_envmap[0] #(H, W, C)
    rgb = rearrange(rgb, 'h w c -> c h w', h=H, w=W)
    return rgb
'''
Get the Gaussians based on the ray_o, ray_d, normals_map
Note that these items can not share the same camera paramters to rasterization
'''
def Baking(viewpoint_camera, pc : GaussianModel, pipe, rays_o, rays_d, normals_map, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    else:
        assert False # The visible mask must be re-calculated!

    #Filter
    # print("Line 350", pc.get_anchor.shape)#(N, 3)
    # anchor_position = rearrange(pc.get_anchor, ('n c-> 1 n c'))
    # anchor_position = repeat(anchor_postion, )

    # cos_anchor_ray_d = torch.sum((rays_o - pc.get_anchor)  * rays_d, dim=1, keepdim=True)
    # coarse_mask_threholds = np.cos(30/180*np.pi)
    # coarse_mask = cos_anchor_ray_d > coarse_mask_threholds
    

    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    # 实际上每个位置进行baking出对应的位置
    N_pixels = rays_d.shape[0]
    out_rgb = torch.zeros_like(rays_d)
    out_tint = torch.zeros_like(rays_d)
    out_depth = torch.zeros_like(rays_d)
    

    # 生成一个1到100的整数列表
    numbers = list(range(N_pixels))

    # 随机抽取100个不重复的元素
    random_numbers = random.sample(numbers, 10)

    # print(random_numbers)
    # for i in range(N_pixels):
    for i in random_numbers:
        ro, rd, norm = rays_o[i], rays_d[i], normals_map[i]
        #实际上可以做加速，不需要渲染六个面
        rgb_cubemap = []
        alpha_cubemap = []
        depth_cubemap = []
        tint_cubemap = []
        #直接从Gs中得到albedo和tint
        all_rotations = GenerateCubemapRotation()
        test_all_occupancy = torch.zeros((N_pixels, 6, 32, 32, 4))
        
        individual_visible_mask = None # can be fast and get the mask
        cos_anchor_ray_d = torch.sum((ro - pc.get_anchor)  * rd, dim=1, keepdim=False)
        # print("Test ", i, ro, rd, norm)
        coarse_mask_threholds = np.cos(15/180*np.pi)
        individual_visible_mask = cos_anchor_ray_d > coarse_mask_threholds
        # print("Get ", individual_visible_mask.sum(), pc.get_anchor.shape, cos_anchor_ray_d.shape)

        for r_idx, rotation in enumerate(all_rotations):
            c2w = rotation
            c2w[:3, 3] = ro
            w2c = torch.inverse(c2w)
            T = w2c[:3, 3]
            R = w2c[:3, :3].T
            fake_view_camera = FakeCamera(R, T, H=16, W=16, 
                                           FoVx=math.pi * 0.5, FoVy=math.pi * 0.5)
            out = CustomedRender(fake_view_camera, pc, pipe, bg_color, 
                                    scaling_modifier, individual_visible_mask, retain_grad)

            rgb_cubemap.append(out["render"].permute(1, 2, 0))
            # alpha_cubemap.append(out["alpha"].permute(1, 2, 0))
            # depth_cubemap.append(out["depth"].permute(1, 2, 0))
            tint_cubemap.append(out["tint"].permute(1, 2, 0))
            norm = norm.reshape((1, 1, 1, 3))
            torch.cuda.empty_cache()
        rgb_envmap = dr.texture(torch.stack(rgb_cubemap)[None, ...],  # [1, 6, 16, 16, 3]
                            norm.contiguous(),  # [1, H, W, 3]
                            filter_mode="linear",
                            boundary_mode="cube",)
        rgb = rgb_envmap[0,0,0,:]
        del rgb_envmap
        del out
        torch.cuda.empty_cache()
        # Save GPU space
        # depth_envmap = dr.texture(torch.stack(depth_cubemap)[None, ...],  # [1, 6, 16, 16, 3]
        #                     norm.contiguous(),  # [1, H, W, 3]
        #                     filter_mode="linear",
        #                     boundary_mode="cube",)
        # depth = depth_envmap[0,0,0,:]
            
        # tint_envmap = dr.texture(torch.stack(tint_cubemap)[None, ...],  # [1, 6, 16, 16, 3]
        #                     norm.contiguous(),  # [1, H, W, 3]
        #                     filter_mode="linear",
        #                     boundary_mode="cube",)
        # tint = tint_envmap[0,0,0,:]
            
        out_rgb[i] = rgb
        # out_depth[i] = depth
        # out_tint[i] = tint
    # assert 1==0
    out_rgb = out_rgb.reshape((H, W, 3))
    out_rgb = rearrange(out_rgb, 'h w c -> c h w', h=H, w=W)
    # out_depth = out_depth.reshape((H, W, 3))
    # out_tint = out_tint.reshape((H, W, 3))
    return out_rgb#, out_depth, out_tint





def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
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

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
