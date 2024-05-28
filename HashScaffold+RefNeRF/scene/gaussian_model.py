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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from scene.NVDIFFREC import create_trainable_env_rnd, load_env
import tinycudann as tcnn
import scene.ref_utils as ref_utils
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.roughness_activation = torch.nn.functional.softplus


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 env_dict: dict={"mode":"mlp_env"},
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        # Modification
        self._normals = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        # Modification
        self.mlp_normals = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets), # Three channel
            nn.Tanh()
        ).cuda()

        # self.mlp_env = nn.Sequential(
        #     nn.Linear(3, feat_dim//2),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim//2, feat_dim), # Feat channel
        # ).cuda()
        # self.mlp_env = nn.Sequential(
        #     nn.Linear(feat_dim + 72 + 1, feat_dim//2),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim//2, 3), # Feat channel
        #     # nn.ReLU(True)
        # ).cuda()
        self.mlp_env = ref_utils.views_mlp(feat_dim).cuda()

        self.mlp_tint = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid(),
        ).cuda()

        self.mlp_roughness = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.ReLU(),#RefNeRF
        ).cuda()

        self.mlp_mentalic = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid(),
        ).cuda()
        self.env_dict = env_dict
        # if env_dict["mode"] == "envmap":
        self.env_res = 512
        # self.mlp_env = create_trainable_env_rnd(self.env_res, scale=0.0, bias=0.8)
        self._env_color = torch.empty(0) # [1, 6, 16, 16, 3]
        
        self.brdf = True
        if self.brdf:
            self.brdf_mlp = create_trainable_env_rnd(self.env_res, scale=0.0, bias=0.8)
        else:
            self.brdf_mlp = None  

        self.use_hash = True
        if self.use_hash:

            # Parameters
            # n_levels = 4
            # base_resolution = 16
            # log2_hashmap_size = 19
            # constants
            L = 16; F = 2; log2_T = 19; N_min = 16
            # per_level_scale = (max_res / N_min) ** (1 / (L - 1))
            per_level_scale = np.exp(np.log(2048*1/N_min)/(L-1))
            print(f'GridEncoding: Nmin={N_min} b={per_level_scale:.5f} F={F} T=2^{log2_T} L={L}')
            n_features_per_level = feat_dim // L
            
            self.hash_config = {
                "otype": "Grid",
	            "type": "Hash",
                "n_levels": L,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"
            }

            self.mlp_anchor_feature = tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=feat_dim,
                encoding_config=self.hash_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            ).cuda()
            # Separate define
            # self.anchor_encoding = tcnn.Encoding(
            #     n_input_dims=3,
            #     encoding_config=self.hash_config,
            # ).cuda()

            # self.anchor_feature_network = nn.Sequential(
            #     nn.Linear(feat_dim, 32),
            #     nn.ReLU(True),
            #     nn.Linear(32, feat_dim),
            #     nn.Tanh(),
            # ).cuda()

            # mlp_anchor_feature = torch.nn.Sequential(anchor_encoding, anchor_feature_network)
        
        self.deg_view = 5
        self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

        self.mlp_normals.eval()
        self.mlp_env.eval()
        self.mlp_tint.eval()
        self.mlp_roughness.eval()
        self.mlp_mentalic.eval()
        self.mlp_anchor_feature.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()

        self.mlp_normals.train()
        self.mlp_env.train()
        self.mlp_tint.train()
        self.mlp_roughness.train()
        self.mlp_mentalic.train()
        self.mlp_anchor_feature.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._env_color,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._env_color,
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    @property
    def get_env_mlp(self):
        return self.mlp_env
    @property
    def get_tint_mlp(self):
        return self.mlp_tint
    @property
    def get_roughness_mlp(self):
        return self.mlp_roughness
    @property
    def get_mentalic_mlp(self):
        return self.mlp_mentalic
    @property
    def get_anchor_feature(self):
        return self.mlp_anchor_feature(self._anchor)
    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_normals_mlp(self):
        return self.mlp_normals
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_env_color(self):
        return self._env_color
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_roughness(self, x):
        return self.roughness_activation(self.mlp_roughness(x) + (-1.0))
    def get_ref_diffuse(self, x):
        return self.opacity_activation(self.mlp_color(x) - np.log(3)) # Reference RefNeRF
    def get_ref_specular(self, x):
        bias = 0
        multiply = 1.0
        return self.opacity_activation(self.mlp_env(x) * multiply + bias) # Reference RefNeRF
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        env_color_map = torch.zeros((1, 6, self.env_res, self.env_res, 3)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        # anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        # modification
        anchors_feat = torch.rand((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        anchors_feat = (anchors_feat - 0.5) * 2

        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # Initialize to the new env map
        self._env_color = nn.Parameter(env_color_map.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        
        
        
        # Modification
        # normals = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3), device="cuda")
        # self._normals = nn.Parameter(normals.requires_grad_(True))
        # print("In model 318 Line opacity shape", self._normals.shape)
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._env_color], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "env_color"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # {'params': [self._normals], 'lr': training_args.normals_lr, "name": "normals"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                # Modification
                {'params': self.mlp_normals.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_normals"},
                {'params': self.mlp_env.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_env"},
                {'params': self.mlp_tint.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_tint"},
                {'params': self.mlp_roughness.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_roughness"},
                {'params': self.mlp_mentalic.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_mentalic"},
                {'params': self.mlp_anchor_feature.parameters(), 'lr': training_args.feature_lr, "name": "mlp_anchor_feature"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._env_color], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "env_color"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # {'params': [self._normals], 'lr': training_args.normals_lr, "name": "normals"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                # Modification
                {'params': self.mlp_normals.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_normals"},
                {'params': self.mlp_env.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_env"},
                {'params': self.mlp_tint.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_tint"},
                {'params': self.mlp_roughness.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_roughness"},
                {'params': self.mlp_mentalic.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_mentalic"},
                {'params': self.mlp_anchor_feature.parameters(), 'lr': training_args.feature_lr, "name": "mlp_anchor_feature"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._env_color], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "env_color"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # {'params': [self._normals], 'lr': training_args.normals_lr, "name": "normals"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                # Modification
                {'params': self.mlp_normals.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_normals"},
                {'params': self.mlp_env.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_env"},
                {'params': self.mlp_tint.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_tint"},
                {'params': self.mlp_roughness.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_roughness"},
                {'params': self.mlp_mentalic.parameters(), 'lr': training_args.mlp_env_lr_init, "name": "mlp_mentalic"},
                {'params': self.mlp_anchor_feature.parameters(), 'lr': training_args.feature_lr, "name": "mlp_anchor_feature"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        if self.brdf:
            l.extend([
                {'params': list(self.brdf_mlp.parameters()), 'lr': training_args.mlp_env_lr_init, "name": "brdf_mlp"},
            ])
        self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_env_lr_init,
                                                    lr_final=training_args.mlp_env_lr_final,
                                                    lr_delay_mult=training_args.mlp_env_lr_delay_mult,
                                                    max_steps=training_args.mlp_env_lr_max_steps)


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        
        self.mlp_normals_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_normals_lr_init,
                                                    lr_final=training_args.mlp_normals_lr_final,
                                                    lr_delay_mult=training_args.mlp_normals_lr_delay_mult,
                                                    max_steps=training_args.mlp_normals_lr_max_steps)
        self.mlp_env_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_env_lr_init,
                                                    lr_final=training_args.mlp_env_lr_final,
                                                    lr_delay_mult=training_args.mlp_env_lr_delay_mult,
                                                    max_steps=training_args.mlp_env_lr_max_steps)
        self.mlp_tint_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_env_lr_init,
                                                    lr_final=training_args.mlp_env_lr_final,
                                                    lr_delay_mult=training_args.mlp_env_lr_delay_mult,
                                                    max_steps=training_args.mlp_env_lr_max_steps)

        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            # Modification
            if param_group["name"] == "mlp_normals":
                lr = self.mlp_normals_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_env":
                lr = self.mlp_env_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_tint":
                lr = self.mlp_tint_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.brdf and param_group["name"] == "brdf_mlp":
                lr = self.brdf_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # Modification
        # for i in range(self._normals.shape[1]):
        #     l.append('f_normals_{}'.format(i))
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        self.save_env_color(path.replace("point_cloud.ply", "env_color.npy"))

    
    def save_env_color(self, path):
        env_color = self._env_color.detach().cpu().numpy()
        np.save(path, env_color)
    # def load_env_color(self, path):
    #     env_color = np.load(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        normals_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_normals")]
        normals_names = sorted(normals_names, key = lambda x: int(x.split('_')[-1]))
        normals = np.zeros((anchor.shape[0], len(normals_names)))
        for idx, attr_name in enumerate(normals_names):
            normals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        normals = normals.reshape((normals.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


        self._normals = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))

        env_color = np.load(path.replace("point_cloud.ply", "env_color.npy"))#(1, 6, res, res, 3)
        self._env_color = nn.Parameter(torch.tensor(env_color, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'env_color' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # print(group["params"], group["params"][0].shape, extension_tensor.shape)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'env_color' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self._normals = optimizable_tensors["normals"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                # print(remove_duplicates_list)
                # import pdb;pdb.set_trace()
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                # new_normals = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()


                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    # "normals": new_normals
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]

                # self._normals = optimizable_tensors["normals"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            self.mlp_normals.eval()
            normal_mlp = torch.jit.trace(self.mlp_normals, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            normal_mlp.save(os.path.join(path, 'normals_mlp.pt'))
            self.mlp_normals.train()

            self.mlp_env.eval()
            env_mlp = torch.jit.trace(self.mlp_env, (torch.rand(1, self.feat_dim+72+1).cuda()))
            env_mlp.save(os.path.join(path, 'env_mlp.pt'))
            self.mlp_env.train()

            self.mlp_tint.eval()
            tint_mlp = torch.jit.trace(self.mlp_tint, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            tint_mlp.save(os.path.join(path, 'tint_mlp.pt'))
            self.mlp_tint.train()

            self.mlp_roughness.eval()
            roughness_mlp = torch.jit.trace(self.mlp_roughness, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            roughness_mlp.save(os.path.join(path, 'roughness_mlp.pt'))
            self.mlp_roughness.train()


            self.mlp_mentalic.eval()
            roughness_mlp = torch.jit.trace(self.mlp_mentalic, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            roughness_mlp.save(os.path.join(path, 'mentalic_mlp.pt'))
            self.mlp_mentalic.train()

            self.mlp_anchor_feature.eval()
            # anchor_feature_mlp = torch.jit.trace(self.mlp_anchor_feature, (torch.rand(1, 3).cuda()))
            # anchor_feature_mlp.save(os.path.join(path, 'anchor_feature_mlp.pt'))

            torch.save(self.mlp_anchor_feature.state_dict(), os.path.join(path, 'anchor_feature_mlp.pt'))
            self.mlp_anchor_feature.train()
            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'normals_mlp': self.mlp_normals.state_dict(),
                    'env_mlp': self.mlp_env.state_dict(),
                    'tint_mlp': self.mlp_tint.state_dict(),
                    'roughness_mlp': self.mlp_roughness.state_dict(),
                    'mentalic_mlp': self.mlp_mentalic.state_dict(),
                    'anchor_feature_mlp': self.mlp_anchor_feature.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'normals_mlp': self.mlp_normals.state_dict(),
                    'env_mlp': self.mlp_env.state_dict(),
                    'tint_mlp': self.mlp_tint.state_dict(),
                    'roughness_mlp': self.mlp_roughness.state_dict(),
                    'mentalic_mlp': self.mlp_mentalic.state_dict(),
                    'anchor_feature_mlp': self.mlp_anchor_feature.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'normals_mlp': self.mlp_normals.state_dict(),
                    'env_mlp': self.mlp_env.state_dict(),
                    'tint_mlp': self.mlp_tint.state_dict(),
                    'roughness_mlp': self.mlp_roughness.state_dict(),
                    'mentalic_mlp': self.mlp_mentalic.state_dict(),
                    'anchor_feature_mlp': self.mlp_anchor_feature.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()

            self.mlp_normals = torch.jit.load(os.path.join(path, 'normals_mlp.pt')).cuda()
            self.mlp_env = torch.jit.load(os.path.join(path, 'env_mlp.pt')).cuda()
            
            self.mlp_tint = torch.jit.load(os.path.join(path, 'tint_mlp.pt')).cuda()
            self.mlp_roughness = torch.jit.load(os.path.join(path, 'roughness_mlp.pt')).cuda()
            self.mlp_mentalic = torch.jit.load(os.path.join(path, 'mentalic_mlp.pt')).cuda()
            # self.mlp_anchor_feature = torch.jit.load(os.path.join(path, 'anchor_feature_mlp.pt')).cuda()
            self.mlp_anchor_feature.load_state_dict(torch.load(os.path.join(path, 'anchor_feature_mlp.pt')))
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()

            self.mlp_opacity.train()
            self.mlp_cov.train()
            self.mlp_color.train()
            self.mlp_normals.train()
            self.mlp_env.train()
            self.mlp_tint.train()
            self.mlp_roughness.train()
            self.mlp_mentalic.train()
            self.mlp_anchor_feature.train()
            if self.use_feat_bank:
                self.mlp_feature_bank.train()
            if self.appearance_dim > 0:
                self.embedding_appearance.train()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            self.mlp_normals.load_state_dict(checkpoint['normals_mlp'])
            self.mlp_env.load_state_dict(checkpoint['env_mlp'])
            self.mlp_tint.load_state_dict(checkpoint['tint_mlp'])
            self.mlp_roughness.load_state_dict(checkpoint['roughness_mlp'])
            self.mlp_mentalic.load_state_dict(checkpoint['mentalic_mlp'])
            self.mlp_anchor_feature.load_state_dict(checkpoint['anchor_feature_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
    
    def post_check(self, cams, pipe, background, prefilter_voxel=None, render=None):
        # iterate over all poses, delete unused
        self.opacity_accum = torch.zeros_like(self.opacity_accum)

        visit_count = torch.zeros(self.get_anchor.shape[0], 1).to(self.get_anchor.device)

        # print(f"before, visit_count: {visit_count.sum()}, {visit_count.shape}")

        for viewpoint_cam in cams:
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, self, pipe,background)

            retain_grad = False
            render_pkg = render(viewpoint_cam, self, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

            offset_selection_mask, scaling, opacity = render_pkg["selection_mask"], render_pkg["scaling"], render_pkg["neural_opacity"]

            offset_selection_mask = offset_selection_mask.view([-1, self.n_offsets]).sum(dim=-1) # [m, 1]

            mapped_elements = torch.zeros(voxel_visible_mask.shape[0], dtype=torch.bool).to(visit_count.device)
            mapped_elements[voxel_visible_mask] = offset_selection_mask>0

            visit_count[mapped_elements] += 1

        valid_idx = (visit_count>1).squeeze(dim=-1)
        # print(f"after, visit_count: {visit_count.sum()}, {visit_count.shape}")

        # print(f'valid_idx: {valid_idx.sum()}')

        self._anchor = self._anchor[valid_idx]
        self._offset = self._offset[valid_idx]
        self._anchor_feat = self._anchor_feat[valid_idx]
        self._scaling = self._scaling[valid_idx]
        self._rotation = self._rotation[valid_idx]
        self._opacity = self._opacity[valid_idx]

        torch.cuda.empty_cache()