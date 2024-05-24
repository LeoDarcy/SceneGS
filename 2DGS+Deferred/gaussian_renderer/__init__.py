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

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal, depths_to_points
from einops import rearrange, repeat
from pbr import CubemapLight, get_brdf_lut, pbr_shading
import torch
import torch.nn.functional as F
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.brdf :
            # color_delta = None
            # delta_normal_norm = None
            # if pipe.brdf_mode=="envmap":
            #     gb_pos = pc.get_xyz # (N, 3) 
            #     view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 

            #     diffuse   = pc.get_diffuse # (N, 3) 
            #     normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
            #     delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)
            #     specular  = pc.get_specular # (N, 3) 
            #     roughness = pc.get_roughness # (N, 1) 
            #     color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])

            #     colors_precomp = color.squeeze() # (N, 3) 
            #     diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
            #     specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 

            #     if pc.brdf_dim>0:
            #         raise NotImplementedError

            # else:
            #     raise NotImplementedError
            colors_precomp = pc.get_albedo
        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    #all {expected depth, alpha, normal, median depth}
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    
    render_extras = {"roughness": pc.get_roughness, "metallic": pc.get_metallic, "albedo": pc.get_albedo}
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    out_materials = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        if render_extras[k].shape[-1] !=3:
            #Make it to (N, 3)
            target = render_extras[k].repeat(1,3)
            assert target.shape[-1] ==3, target.shape
        image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = target,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        out_materials[k] = image        

    # BJY modification Add deferred rendering
    if pipe.brdf_mode=="envmap":
        pc.brdf_mlp.build_mips()
        depth_image = surf_depth
        # print("check ", depth_image.shape, depth_image[0].shape)
        gd_pos = depths_to_points(viewpoint_camera, depth_image)
        # gd_pos :(h*w, 3)
        # print("check ", gd_pos.shape)
        
        # _, H, W = gd_pos.shape
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
        view_pos = viewpoint_camera.camera_center.repeat(gd_pos.shape[0], 1) # (N, 3) 
        
        # view_dirs = viewpoint_camera.get_rays()
        # gb_pos = viewpoint_camera.camera_center.repeat(view_dirs.shape[0], 1)
        
        
        normal = surf_normal
        normal = rearrange(normal, "c h w -> (h w) c").detach()
        diffuse = rendered_image
        diffuse = rearrange(diffuse, "c h w -> (h w) c") #(N, 3)
        roughness = out_materials["roughness"]# (N, 3) 
        roughness = rearrange(roughness, 'c h w->(h w) c')[:,0:1]
        specular = out_materials["metallic"]# (N, 3) 
        specular = rearrange(specular, 'c h w->(h w) c')
        # print(gb_pos.shape,normal.shape, diffuse.shape, roughness.shape, specular.shape, view_pos.shape)
        # roughness = pc.get_roughness # (N, 1) 
        color, brdf_pkg = pc.brdf_mlp.shade(gd_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])

        colors_deferred = color.squeeze() # (N, 3)
        colors_deferred = rearrange(colors_deferred, '(h w) c -> c h w', h=H, w=W)

        out_materials["color_env"] = colors_deferred
        # + rendered_image# shade do not did this! 
        # diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
        # specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 
    
    
    
    elif pipe.brdf_mode=="GSIR":
        brdf_lut = get_brdf_lut().cuda()
        canonical_rays = get_canonical_rays(viewpoint_camera)
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
        c2w = torch.inverse(viewpoint_camera.world_view_transform.T)  # [4, 4]
        view_dirs = -((F.normalize(canonical_rays[:, None, :], p=2, dim=-1)
                        * c2w[None, :3, :3]
                    )  # [HW, 3, 3]
                    .sum(dim=-1)
                    .reshape(H, W, 3))  # [H, W, 3]
        pc.brdf_mlp.build_mips() # build mip for environment light
        normal_map = render_normal#surf_normal
        albedo_map = rendered_image
        # diffuse = rearrange(diffuse, "c h w -> (h w) c") #(N, 3)
        roughness_map = out_materials["roughness"][0:1, :,:]# (N, 3) 
        metallic_map = out_materials["metallic"]# (1, H, W) 
        # specular = rearrange(specular, 'c h w->(h w) c')
        normal_mask = (normal_map != 0).all(0, keepdim=True)#roughness_map > 0 #torch.ones_like(roughness_map).cuda()
        # print(normal_mask.shape,rets["radii"].shape )
        # normal_mask = rearrange(normal_mask, '(h w)->1 h w', h=viewpoint_camera.image_height, w=viewpoint_camera.image_width)
        pbr_result = pbr_shading(
                light=pc.brdf_mlp,
                normals=normal_map.permute(1, 2, 0).detach(),  # [H, W, 3]
                view_dirs=view_dirs,
                mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
                albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                metallic=metallic_map.permute(1, 2, 0),  # [H, W, 1]
                tone=False,
                gamma=False,
                occlusion=torch.ones_like(roughness_map).permute(1, 2, 0) ,
                irradiance=torch.zeros_like(roughness_map).permute(1, 2, 0),
                brdf_lut=brdf_lut,
            )
        gs_ir_render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
            
    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })
    rets.update(out_materials)
    if pipe.brdf_mode=="envmap":
        rets["render"] = out_materials["color_env"] # + rets["render"] # Has done in light
    elif pipe.brdf_mode=="GSIR":
        rets["render"] = gs_ir_render_rgb
    return rets

#From GS-IR
def get_canonical_rays(ref_camera, scale: float = 1.0) -> torch.Tensor:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        # ref_camera = viewpoint_camera.train_cameras[scale][0]
        # TODO: inject intrinsic
        H, W = ref_camera.image_height, ref_camera.image_width
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(ref_camera.FoVx * 0.5)
        tan_fovy = math.tan(ref_camera.FoVy * 0.5)
        focal_x = W / (2.0 * tan_fovx)
        focal_y = H / (2.0 * tan_fovy)

        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        x = x.flatten()  # [H * W]
        y = y.flatten()  # [H * W]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cen_x + 0.5) / focal_x,
                    (y - cen_y + 0.5) / focal_y,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [H * W, 3]
        # NOTE: it is not normalized
        return camera_dirs.cuda()