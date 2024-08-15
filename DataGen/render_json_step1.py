import json
import math
import os
from dataclasses import dataclass
import random
from typing import Optional

import imageio
import numpy as np
import simple_parsing

import pandas as pd


@dataclass
class Options:
    """ 3D dataset rendering script """
    model_list_json: str = '../assets/objaverse_fids/filtered_fids.txt'  # Base path to 3D models
    model_dir_path: str = '/nas/shared/pjlab_lingjun_landmarks/baijiayang/dataset/Objaverse/glbs/' # dir to model path
    env_map_list_json: str = '../assets/polyhaven_hdris.json'  # Path to env map list
    env_map_dir_path: str = '../hdris/'  # Path to env map directory
    # white_env_map_dir_path: str = './assets/hdri/file_bw'  # Path to white env map directory
    output_dir: str = '../output'  # Output directory
    num_views: int = 2  # Number of views
    num_white_pls: int = 3  # Number of white point lighting
    num_rgb_pls: int = 0  # Number of RGB point lighting
    num_multi_pls: int = 3  # Number of multi point lighting
    max_pl_num: int = 3  # Maximum number of point lights
    num_white_envs: int = 3  # Number of white env lighting
    num_env_lights: int = 3  # Number of env lighting
    num_area_lights: int = 3  # Number of area lights
    seed: Optional[int] = None  # Random seed


def render_core(args: Options):
    import bpy

    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.io import render_depth_map, mat2list, array2list, render_normal_map
    from bpy_helper.light import create_point_light, set_env_light, create_area_light, remove_all_lights
    from bpy_helper.material import create_white_diffuse_material, create_specular_ggx_material, clear_emission_and_alpha_nodes
    from bpy_helper.random import gen_random_pts_around_origin
    from bpy_helper.scene import import_3d_model, normalize_scene, reset_scene
    from bpy_helper.utils import stdout_redirected

    # Import the 3D object
    kiui_uids = pd.read_csv(args.model_list_json, header=None)

    uids = kiui_uids[0]
    # uids = uids.str.split('/').str[1]
    uids = uids.values.tolist()
    print("Load models ", len(uids))
    

    # Load env map list
    env_map_list = json.load(open(args.env_map_list_json, 'r'))

    # Render GT images & hints
    seed_view = None if args.seed is None else args.seed
    
    for uid in uids:
        model_path = args.model_dir_path + uid + '.glb'
        if os.path.isfile(model_path) ==False:
            print("Skip model ", uid)
            continue
        else:
            print("Access ", uid)
        res_dir = f"{args.output_dir}/{uid}"
        os.makedirs(res_dir, exist_ok=True)

        # Check path 
        
        eyes = gen_random_pts_around_origin(
            seed=seed_view,
            N=args.num_views,
            min_dist_to_origin=1.0,
            max_dist_to_origin=1.0,
            min_theta_in_degree=10,
            max_theta_in_degree=90
        )
        for eye_idx in range(args.num_views):
            # 0. Place Camera, render gt depth map
            eye = eyes[eye_idx]
            # fov = random.uniform(25, 35)
            fov = random.uniform(85, 95)
            radius = random.uniform(0.8, 1.1) * (0.5 / math.tanh(fov / 2. * (math.pi / 180.)))
            eye = [x * radius for x in eye]
            c2w = look_at_to_c2w(eye)
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/view_{eye_idx}'
            os.makedirs(view_path, exist_ok=True)
            # assert 1==0, view_path
            # save cam info
            json.dump({'c2w': mat2list(c2w), 'fov': fov}, open(f'{view_path}/cam.json', 'w'), indent=4)

            # 5. Env lighting
            for env_map_idx in range(args.num_env_lights):
                env_map = random.choice(env_map_list)
                env_map_path = f'{args.env_map_dir_path}/{env_map}_hdri_2k.exr'
                while os.path.isfile(env_map_path) ==False:
                    env_map = random.choice(env_map_list)
                    env_map_path = f'{args.env_map_dir_path}/{env_map}_hdri_2k.exr'
                
                rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
                strength = 1.0  # random.uniform(0.8, 1.2)
                env_path = f'{view_path}/env_{env_map_idx}'
                os.makedirs(env_path, exist_ok=True)
                # save env map info
                json.dump({
                    'env_map': env_map,
                    'rotation_euler': rotation_euler,
                    'strength': strength,
                }, open(f'{env_path}/env.json', 'w'), indent=4)
        # assert False


if __name__ == '__main__':
    args: Options = simple_parsing.parse(Options)
    print("options:", args)
    render_core(args)
