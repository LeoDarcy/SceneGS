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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import mkdir_p
from scene.NVDIFFREC import save_env_map, load_env

import torch
import torch.nn.functional as F
import math
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif "eyefultower" in args.source_path or "fipt" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["EyeFul"](args.source_path, random_background=False, white_background=False, eval=args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if self.gaussians.brdf:
                fn = os.path.join(self.model_path,
                                "brdf_mlp",
                                "iteration_" + str(self.loaded_iter),
                                "brdf_mlp.hdr")
                self.gaussians.brdf_mlp = load_env(fn, scale=1.0)
                print(f"Load envmap from: {fn}")
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.brdf:
            brdf_mlp_path = os.path.join(self.model_path, f"brdf_mlp/iteration_{iteration}/brdf_mlp.hdr")
            mkdir_p(os.path.dirname(brdf_mlp_path))
            save_env_map(brdf_mlp_path, self.gaussians.brdf_mlp)
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


    #From GS-IR
    def get_canonical_rays(self, scale: float = 1.0) -> torch.Tensor:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        ref_camera = self.train_cameras[scale][0]
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