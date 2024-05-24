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
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.camera_utils import gen_cam_uv
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import csv
import h5py
import torch
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    normal_image: np.array
    alpha_mask: np.array
    depth_image: np.array
    albedo_image: np.array
    residure_image: np.array
    


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# Borrorw from Hypersim scene_generate_images_tonemap.py

def Tonemapping(rgb_color, out_file=None):
    gamma                             = 1.0/2.2   # standard gamma correction exponent
    inv_gamma                         = 1.0/gamma
    percentile                        = 90        # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8       # ...to be this bright after scaling

    # valid_mask = render_entity_id != -1
    # Suppose that rgb color is all valid? Not sure
    valid_mask  = np.ones_like(rgb_color, dtype=np.uint8)
    # valid_mask = rgb_color != 0

    if np.sum(valid_mask) == 0:
        scale = 1.0 # if there are no valid pixels, then set scale to 1.0
    else:
        brightness       = 0.3*rgb_color[:,:,0] + 0.59*rgb_color[:,:,1] + 0.11*rgb_color[:,:,2] # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps                               = 0.0001 # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:

            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    rgb_color_tm = np.power(np.maximum(scale*rgb_color,0), gamma)

    print("[HYPERSIM: SCENE_GENERATE_IMAGES_TONEMAP] Saving output file: " + out_file + " (scale=" + str(scale) + ")")
    if out_file is not None:
        output_rgb = np.clip(rgb_color_tm,0,1)
        output_rgb = Image.fromarray(np.array(output_rgb*255.0, dtype=np.byte), "RGB")
        output_rgb.save(out_file)

# Borrow from Relightable Gaussian tonemapping
# Tone Mapping
def Aces_film_Tonemap(rgb, out_file=None) :
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        rgb = rgb.clip(min=0.0, max=1.0)
        if out_file is not None:
            output_rgb = Image.fromarray(np.array(rgb*255.0, dtype=np.byte), "RGB")
            output_rgb.save(out_file)
    elif isinstance(rgb, torch.Tensor):
        rgb = rgb.clamp(min=0.0, max=1.0)
    return rgb

# From https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py
import math
    
def Snavely_Tonemap(rgb, out_file=None) :
    img = rgb
    # Set the median pixel to the median point (default 0.5),
    # (after 2.2).
    # Ideas: Try mapping 90 %-tile to 0.9?
    # Detect overexposure and only underexpose in that case?
    # Some magic number
    FLAGS_percentile = 70 # Smaller-> brighter
    FLAGS_percentile_point = 0.5
    brightness = 0.3 * img[:,:,0] + 0.59 * img[:,:,1] + 0.11 * img[:,:,2]
    median = np.median(brightness)
    percentile = np.percentile(brightness, FLAGS_percentile)
    if median < 1.0e-4:
        scale = 0.0
    else:
        scale = math.exp(math.log(FLAGS_percentile_point) * 2.2 - math.log(percentile))


    img = np.clip(np.power(scale * img, 1/2.2), 0,1)
    if out_file is not None:
        output_rgb = Image.fromarray(np.array(img*255.0, dtype=np.byte), "RGB")
        output_rgb.save(out_file)
    return img


# Attention that our output is (0,1) not (0,255)

def readHyperSimInfo(path, white_background, eval, llffhold=8, extension=".png", ply_path=None, Tonemap="Snavely"):
    tonemap_dic = {
        "Snavely":Snavely_Tonemap,
        "Apple":Tonemapping,
        "Aces_film_Tonemap":Aces_film_Tonemap
    }
    tonemap_func = tonemap_dic[Tonemap]
    # Prepare for check tonemapping
    BJY_SAVE = False
    if BJY_SAVE:
        BJY_save_root = '/cpfs01/user/baijiayang/workspace/workspace/Code/HypersimDataset/Scaffold-GS_EnvTanh/outputs/checkdataset'
        print(Fore.RED + "Attention !!! Saving all dataset image in " + Fore.YELLOW + BJY_save_root + Style.RESET_ALL)
    
    hypersim_path = "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads/"
    csv_filename = os.path.join(hypersim_path, "metadata_images_split_scene_v1.csv")
    assert os.path.exists(csv_filename)

    # read the csv file first
    with open(csv_filename, encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        metadata = {}
        for row in reader:
            for column, value in row.items():
                metadata.setdefault(column, []).append(value)

    # not only train
    included_in_public_release = np.array(metadata["included_in_public_release"])
    public_index = included_in_public_release == "True"
    split_partitions = np.array(metadata["split_partition_name"])[public_index]
    scene_names = np.array(metadata["scene_name"])[public_index]
    camera_names = np.array(metadata["camera_name"])[public_index]
    frame_ids = np.array(metadata["frame_id"])[public_index].astype(np.int64)
    # scene_name_unique = np.unique(scene_names)

    # read cameras
    camera_filename = os.path.join(hypersim_path, "metadata_camera_parameters.csv")
    assert os.path.exists(camera_filename)
    with open(camera_filename, encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        camera_metadata = {}
        for row in reader:
            for column, value in row.items():
                camera_metadata.setdefault(column, []).append(value)
    scene_name = path.split("/")[-1]

    scene_mask = scene_names == scene_name
    camera_names_perscene = camera_names[scene_mask]
    frame_ids_perscene = frame_ids[scene_mask]
    split_perscene = split_partitions[scene_mask]

    i_ = camera_metadata["scene_name"].index(scene_name)
    width_pixels = int(
        np.round(float(camera_metadata["settings_output_img_width"][i_]))
    )
    height_pixels = int(
        np.round(float(camera_metadata["settings_output_img_height"][i_]))
    )
    meters_per_asset_unit = float(
        camera_metadata["settings_units_info_meters_scale"][i_]
    )
    key_list = [
        "M_cam_from_uv_00",
        "M_cam_from_uv_01",
        "M_cam_from_uv_02",
        "M_cam_from_uv_10",
        "M_cam_from_uv_11",
        "M_cam_from_uv_12",
        "M_cam_from_uv_20",
        "M_cam_from_uv_21",
        "M_cam_from_uv_22",
    ]
    uv2c = (
        np.array([camera_metadata[n_][i_] for n_ in key_list])
        .astype(np.float32)
        .reshape(3, 3)
    )

    key_list = [
        "M_proj_00",
        "M_proj_01",
        "M_proj_02",
        "M_proj_03",
        "M_proj_10",
        "M_proj_11",
        "M_proj_12",
        "M_proj_13",
        "M_proj_20",
        "M_proj_21",
        "M_proj_22",
        "M_proj_23",
        "M_proj_30",
        "M_proj_31",
        "M_proj_32",
        "M_proj_33",
    ]
    M_proj = (
        np.array([camera_metadata[n_][i_] for n_ in key_list])
        .astype(np.float32)
        .reshape(4, 4)
    )

    fl_x = M_proj[0, 0] * width_pixels / 2
    fl_y = M_proj[1, 1] * height_pixels / 2

    c2ws = []
    depths = []
    images = [] # aggregate tonemap
    albedos = []
    normals = []
    cam_frame_split = []
    cam_infos = []

    ct = 0

    print("total: " + Fore.RED + scene_name + Fore.YELLOW + str(len(frame_ids_perscene)) + Style.RESET_ALL)

    for cam_name in np.unique(camera_names_perscene):
        camera_pos_hdf5 = os.path.join(
            hypersim_path,
            scene_name,
            "_detail",
            cam_name,
            "camera_keyframe_positions.hdf5",
        )
        camera_c2w_hdf5 = os.path.join(
            hypersim_path,
            scene_name,
            "_detail",
            cam_name,
            "camera_keyframe_orientations.hdf5",
        )

        with h5py.File(camera_pos_hdf5, "r") as f:
            camera_poss = f["dataset"][:]
        with h5py.File(camera_c2w_hdf5, "r") as f:
            camera_c2ws = f["dataset"][:]

        cam_mask = camera_names_perscene == cam_name
        frame_ids_percam = frame_ids_perscene[cam_mask]
        split_percam = split_perscene[cam_mask]

        for frame_i, split_i in zip(frame_ids_percam, split_percam):
            depths_meters_hdf5 = os.path.join(
                hypersim_path,
                scene_name,
                "images",
                f"scene_{cam_name}_geometry_hdf5",
                f"frame.{int(frame_i):04d}.depth_meters.hdf5",
            )
            with h5py.File(depths_meters_hdf5, "r") as f:
                hypersim_depth_meters = f["dataset"][:].astype(np.float32)

            # Normals
            normal_hdf5 = os.path.join(
                hypersim_path,
                scene_name,
                "images",
                f"scene_{cam_name}_geometry_hdf5",
                f"frame.{int(frame_i):04d}.normal_world.hdf5",
            )
            with h5py.File(normal_hdf5, "r") as f:
                hypersim_normals = f["dataset"][:].astype(np.float32)
                hyper_norm = np.linalg.norm(x=hypersim_normals, ord=2, axis=2, keepdims=True)
                hypersim_normals /= hyper_norm
            normals.append(hypersim_normals)
            

            reflectance_postfix = f"frame.{int(frame_i):04d}.diffuse_reflectance"
            illumination_postfix = f"frame.{int(frame_i):04d}.diffuse_illumination"
            residual_postfix = f"frame.{int(frame_i):04d}.residual"

            image_hdf5 = os.path.join(
                hypersim_path, scene_name, "images", f"scene_{cam_name}_final_hdf5"
            )

            with h5py.File(
                os.path.join(image_hdf5, reflectance_postfix + ".hdf5"), "r"
            ) as f:
                reflectance_hdf5 = f["dataset"][:].astype(np.float32)
            with h5py.File(
                os.path.join(image_hdf5, illumination_postfix + ".hdf5"), "r"
            ) as f:
                illumination_hdf5 = f["dataset"][:].astype(np.float32)
            with h5py.File(
                os.path.join(image_hdf5, residual_postfix + ".hdf5"), "r"
            ) as f:
                residual_hdf5 = f["dataset"][:].astype(np.float32)
            # illumination_hdf5 = to_pq(illumination_hdf5, source="linear")

            aggregate_hdf5 = (reflectance_hdf5 * illumination_hdf5) + residual_hdf5
            if BJY_SAVE:
                aggregate_hdf5 = tonemap_func(aggregate_hdf5, out_file=os.path.join(BJY_save_root, f"{int(frame_i):04d}_aggregate_tonemap.jpg"))
                
            else:
                aggregate_hdf5 = tonemap_func(aggregate_hdf5)

            image = Image.fromarray(np.uint8(aggregate_hdf5 * 255)) 
            images.append(image)

            
            albedos.append(reflectance_hdf5)

            #Two ways to get gt color: load hdr color and tonemapping or directly load SCENE_NAME/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg
            gt_color_postfix = f"frame.{int(frame_i):04d}.color"
            with h5py.File(
                os.path.join(image_hdf5, gt_color_postfix + ".hdf5"), "r"
            ) as f:
                gt_color_hdf5 = f["dataset"][:].astype(np.float32)
            if BJY_SAVE:
                # gt_color_hdf5 = Tonemapping(gt_color_hdf5, out_file=os.path.join(BJY_save_root, f"{int(frame_i):04d}_tonemap.jpg"))
                gt_color_hdf5 = tonemap_func(gt_color_hdf5, out_file=os.path.join(BJY_save_root, f"{int(frame_i):04d}_tonemap.jpg"))
            else:
                # gt_color_hdf5 = Tonemapping(gt_color_hdf5, out_file=os.path.join(BJY_save_root))
                gt_color_hdf5 = tonemap_func(gt_color_hdf5)
            image_RGB = os.path.join(
                hypersim_path, scene_name, "images", f"scene_{cam_name}_final_preview"
            )
            gt_color_postfix = f"frame.{int(frame_i):04d}.tonemap"
            gt_color_RGB = Image.open(os.path.join(image_RGB, gt_color_postfix + ".jpg"))
            if BJY_SAVE:
                gt_color_RGB.save(os.path.join(BJY_save_root, f"{int(frame_i):04d}_AppleGT.jpg"))


            camera_pos = camera_poss[frame_i]
            camera_c2w = camera_c2ws[frame_i]
            extrin = np.eye(4)
            extrin[:3, :3] = camera_c2w
            extrin[:3, 3] = camera_pos * meters_per_asset_unit
            depths.append(hypersim_depth_meters)

            c2ws.append(extrin)
            cam_frame_split.append((cam_name, frame_i, split_i))

            c2w = extrin.copy()
            c2w_T = c2w[:3, -1]
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = image_hdf5
            image_name = Path(image_path).stem

            FovY = focal2fov(fl_y, height_pixels)
            FovX = focal2fov(fl_x, width_pixels)

            # width, height = image.size

            # Attention image must be PIL
            # But normal image and albedo image must be tensor (C, H,  W)
            # Now change them into tensor
            albedo_tensor = torch.from_numpy(np.array(reflectance_hdf5)).permute(2, 0, 1)
            normals_tensor = torch.from_numpy(np.array(hypersim_normals)).permute(2, 0, 1)
            cam_infos.append(
                CameraInfo(
                    uid=ct,
                    R=R,
                    T=T,
                    # c2w_T=c2w_T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width_pixels,
                    height=height_pixels,
                    normal_image=normals_tensor,
                    alpha_mask=None,
                    depth_image=hypersim_depth_meters,
                    albedo_image=albedo_tensor,
                    residure_image=residual_hdf5
                )
            )

            ct += 1


    # assert False, "Check point here"
    
    # collect all frames Not need depths
    depths_npy = np.stack(depths)
    # c2ws_npy = np.stack(c2ws)

    depth_sorted_order = np.argsort(
        depths_npy.reshape(depths_npy.shape[0], -1).mean(-1)
    )

    uv2c = torch.tensor(uv2c).float().cuda()
    depths = [torch.tensor(d).float().cuda() for d in depths]

    c2ws = [torch.tensor(c).float().cuda() for c in c2ws]
    num_frames = len(cam_frame_split)

    print("Load Cameras: ", len(cam_infos))
    train_cam_infos = []

    if not eval:
        train_cam_infos.extend(cam_infos)
        train_idx = np.range(len(cam_infos))
        test_cam_infos = []
    else:
        train_idx = [idx for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_idx = [idx for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        

    # point cloud initialization
    if ply_path is None:
        ply_path = './tmp.ply'
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        nerf_normalization = getNerfppNorm(train_cam_infos)
        
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif not os.path.exists(ply_path):
        pt3ds = None
        rgbs = []
        
        sample_ratio = 10
        for i in train_idx:
            depth_i = depths[i]
            image_i = images[i]
            c2w_i = c2ws[i]

            nan_ma = torch.isnan(depth_i)
            nan_ratio = torch.count_nonzero(nan_ma).item() / (width_pixels * height_pixels)

            
            uv = gen_cam_uv(width_pixels, height_pixels).reshape(-1, 3)
            uv = torch.tensor(uv).float().cuda()
            pt3d = uv2c @ uv.T

            pt3d = pt3d / torch.norm(pt3d, dim=0, keepdim=True)
            pt3d = pt3d * depth_i.reshape(1, -1)
            pt3d = c2w_i[:3, :3] @ pt3d + c2w_i[:3, 3:4]

            # filter out nan depth ratio
            # pt3ds.append(
            #     pt3d.T.reshape(height_pixels, width_pixels, 3)[~nan_ma][::sample_ratio]
            # )
            if pt3ds is None:
                pt3ds = pt3d.T.reshape(height_pixels, width_pixels, 3)[~nan_ma][::sample_ratio]
            else:
                pt3ds =torch.cat([pt3ds, pt3d.T.reshape(height_pixels, width_pixels, 3)[~nan_ma][::sample_ratio]], dim=0)
            rgbs.append(np.array(image_i)[~nan_ma.cpu().numpy()][::sample_ratio])

        # pt3ds = torch.cat(pt3ds).cpu().numpy()
        pt3ds = pt3ds.cpu().numpy()
        rgbs = np.concatenate(rgbs) / 255.0

        nerf_normalization = getNerfppNorm(train_cam_infos)
        normals = np.zeros_like(pt3ds)

        pcd = BasicPointCloud(points=pt3ds, colors=rgbs, normals=normals)

        # ply_path = os.path.join(path, "points3d.ply")
        storePly(ply_path, pt3ds, rgbs * 255)
        print("We have saved point cloud", ply_path)
        tmp_ply_path = ply_path.replace('.ply','_temp.ply')
        if not os.path.exists(tmp_ply_path):

            import open3d as o3d
            # 读取点云数据
            pcd = o3d.io.read_point_cloud(ply_path)

            # 进行体素下采样
            down_pcd = pcd.voxel_down_sample(voxel_size=0.5)
            # tmp_ply_path = ply_path.replace('.ply','_temp.ply')
            o3d.io.write_point_cloud(tmp_ply_path, down_pcd)
        

            print("Now we downsample point cloud from "+Fore.RED + str(len(pcd.points))+ " to "+ Fore.YELLOW + str(len(down_pcd.points)) + Style.RESET_ALL)
        pcd = fetchPly(tmp_ply_path)
    else:
        # try:
        print(f'start fetching data from ply file')
        nerf_normalization = getNerfppNorm(train_cam_infos)
        # pcd = fetchPly(ply_path)
        # Before fetch, we downsample points cloud to avoid OOM
        tmp_ply_path = ply_path.replace('.ply','_temp.ply')
        if not os.path.exists(tmp_ply_path):

            import open3d as o3d
            # 读取点云数据
            pcd = o3d.io.read_point_cloud(ply_path)

            # 进行体素下采样
            down_pcd = pcd.voxel_down_sample(voxel_size=0.5)#voxel_size=0.05
            # tmp_ply_path = ply_path.replace('.ply','_temp.ply')
            o3d.io.write_point_cloud(tmp_ply_path, down_pcd)
        

            print("Now we downsample point cloud from "+Fore.RED + str(len(pcd.points))+ " to "+ Fore.YELLOW + str(len(down_pcd.points)) + Style.RESET_ALL)
        pcd = fetchPly(tmp_ply_path)
    

    # print("saved to ", ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
