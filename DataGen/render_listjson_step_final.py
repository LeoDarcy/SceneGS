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
import cv2
from utils import get_mask
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
    cnt = 0
    items_list = []
    for uid in uids:
        cnt += 1
        # if cnt >630:
        #     break
        model_path = args.model_dir_path + uid + '.glb'
        if os.path.isfile(model_path) ==False:
            print("Skip model ", uid)
            continue
        else:
            print(f"Access {cnt}", uid)
        res_dir = f"{args.output_dir}/{uid}"
        os.makedirs(res_dir, exist_ok=True)

        # Check path 
        for eye_idx in range(args.num_views):
            # 0. Place Camera, render gt depth map
            view_path = f'{res_dir}/view_{eye_idx}'
            contents = json.load(open(f'{view_path}/cam.json', 'r'))

            for env_map_idx in range(args.num_env_lights):
                env_path = f'{view_path}/env_{env_map_idx}'

                env_contents = json.load(open(f'{env_path}/env.json', 'r'))
                env_map = env_contents["env_map"]
                env_map_path = f'{args.env_map_dir_path}/{env_map}_hdri_2k.exr'
                assert os.path.isfile(env_map_path), env_map

                # Prepare single item
                gt = f'{env_path}/gt_shadow.exr'
                gt_img = f'{env_path}/gt.png'
                refl = f'{env_path}/gt_refl_01.png'
                ao = f'{env_path}/gt_AO.png'
                mask = f'{env_path}/gt_obj_mask.png'
                input_plane =  f'{env_path}/gt_noObj.png'
                
                item = {"gt":gt, "gt_img":gt_img, "refl":refl, "ao":ao, "mask":mask, "env":env_map_path, "input_plane":input_plane}
                for it in item:
                    assert os.path.isfile(item[it]), item[it] 
                # load input 
                input_path = f'{env_path}/input.png'
                if os.path.isfile(input_path) == False:

                    # Use the next env rendering to get the fake composited image
                    next_env = (env_map_idx + 1) % args.num_env_lights
                    next_obj = cv2.imread(f'{view_path}/env_{next_env}/gt.png')
                    current_plane = cv2.imread(input_plane)

                    current_mask = cv2.imread(mask) 
                    obj_mask, plane_mask = get_mask(current_mask) #(H, W, 1)
                    obj_mask = np.repeat(obj_mask, 3, axis=2)
                    current_plane[obj_mask] = next_obj[obj_mask]
                    cv2.imwrite(input_path, current_plane)
                item["input"] = input_path
                items_list.append(item)
    json.dump({'items': items_list}, open(f'train_list.json', 'w'), indent=4)

if __name__ == '__main__':
    args: Options = simple_parsing.parse(Options)
    print("options:", args)
    render_core(args)
