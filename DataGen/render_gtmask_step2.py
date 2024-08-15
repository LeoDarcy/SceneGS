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
    from bpy_helper.scene import import_3d_model, normalize_scene, reset_scene, scene_bbox
    from bpy_helper.utils import stdout_redirected

    def render_rgb(output_path):
        # bpy.context.scene.view_layers["ViewLayer"].material_override = None
        bpy.context.scene.render.image_settings.file_format = 'PNG'  # set output to png (with tonemapping)
        bpy.context.scene.render.filepath = f'{output_path}.png'
        bpy.ops.render.render(animation=False, write_still=True)
        # img = imageio.v3.imread(f'{output_path}.png') / 255.
        # if img.shape[-1] == 4:
        #     img = img[..., :3] * img[..., 3:]  # fix edge aliasing
        # imageio.v3.imwrite(f'{output_path}.png', (img * 255).clip(0, 255).astype(np.uint8))

        # MAT_DICT = {
        #     '_diffuse': create_white_diffuse_material(),
        #     '_ggx0.05': create_specular_ggx_material(0.05),
        #     '_ggx0.13': create_specular_ggx_material(0.13),
        #     '_ggx0.34': create_specular_ggx_material(0.34),
        # }

        # # render
        # for mat_name, mat in MAT_DICT.items():
        #     bpy.context.scene.view_layers["ViewLayer"].material_override = mat
        #     bpy.context.scene.render.filepath = f'{output_path}{mat_name}.png'
        #     bpy.ops.render.render(animation=False, write_still=True)
        #     img = imageio.v3.imread(f'{output_path}{mat_name}.png') / 255.
        #     if img.shape[-1] == 4:
        #         img = img[..., :3] * img[..., 3:]  # fix edge aliasing
        #     imageio.v3.imwrite(f'{output_path}{mat_name}.png', (img * 255).clip(0, 255).astype(np.uint8))
    def reset_materials():
        # # bpy.context.scene.render.shading_rate = 0
        # remove_all_lights()
        

        # 获取所有物体
        all_objects = bpy.data.objects

        # 遍历所有物体
        for obj in all_objects:
            if obj.type =='CAMERA':
                continue
            # 尝试获取物体的材质，如果没有则创建新的材质
            try:
                material = obj.data.materials
            except:
                continue
                material = bpy.data.materials.new(name="EmissionMaterial")
                obj.data.materials.append(material)
            # 确保物体有材质
            if obj.data.materials:
                material = obj.data.materials[0]
            else:
                # 创建新的材质
                material = bpy.data.materials.new(name="EmissionMaterial")
                obj.data.materials.append(material)
            
            # 确保材质使用节点
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # 清除所有现有节点
            for node in nodes:
                nodes.remove(node)
             # 创建新的节点
            emission_node = nodes.new(type='ShaderNodeEmission')
            # bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            output_node = nodes.new(type='ShaderNodeOutputMaterial')

            # 设置发射节点的颜色为黑色
            
            emission_node.inputs['Color'].default_value = (1, 0, 0, 1)
            if obj.name == "Plane":
                emission_node.inputs['Color'].default_value = (0, 1, 0, 1)
            # 连接节点
            # links.new(emission_node.outputs['Emission'], bsdf_node.inputs['Emission'])
            # links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
            links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
        
        # bpy.context.scene.render.image_settings.file_format = 'PNG'  # set output to png (with tonemapping)
        # bpy.context.scene.render.filepath = f'{output_path}_obj_mask.png'
        # bpy.ops.render.render(animation=False, write_still=True)
    def my_remove_added_objects() -> None:
        """
        Removes the default objects from the scene.
        """
        # 获取所有物体
        all_objects = bpy.data.objects

        # 定义要保留的物体名称
        reserved_names = ["Cube", "Camera", "Light", "Plane"]

        # 遍历所有物体
        for obj in all_objects:
            if obj.name not in reserved_names:
                bpy.data.objects.remove(obj, do_unlink=True)
    def add_plane():
        # 找底下
        bbox_min, bbox_max = scene_bbox()
        bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, align='WORLD', location=bbox_min)
        bpy.context.view_layer.update()
    

    

    def configure_blender():
        # Set the render resolution
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        # Enable the alpha channel for GT mask
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        # bpy.context.scene.view_settings.view_transform = 'Raw'
        # bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        # bpy.context.view_layer.update()

    # Import the 3D object
    kiui_uids = pd.read_csv(args.model_list_json, header=None)

    uids = kiui_uids[0]
    # uids = uids.str.split('/').str[1]
    uids = uids.values.tolist()
    print("Load models ", len(uids))
    cnt=0
    for uid in uids:
        cnt += 1
        if cnt < 870:
            continue
        # if cnt > 10:
        #     break
        model_path = args.model_dir_path + uid + '.glb'
        if os.path.isfile(model_path) ==False:
            print("Skip model ", uid)
            continue
        else:
            print(f"Access {cnt}", uid)
        
        # os.makedirs(res_dir, exist_ok=True)
        reset_scene()


        # Import the 3D object
        file_path = model_path
        # print("before", bpy.data.objects.keys())
        # This time also add objects
        with stdout_redirected():
            import_3d_model(file_path)
        
        # print("after", bpy.data.objects.keys())
        # print(bpy.data.objects.keys())
        scale, offset = normalize_scene(use_bounding_sphere=False)
        clear_emission_and_alpha_nodes()

        # Configure blender
        configure_blender()
        add_plane()
        reset_materials()
        # Load env map list
        env_map_list = json.load(open(args.env_map_list_json, 'r'))

        # Render GT images & hints
        seed_view = None if args.seed is None else args.seed
        seed_white_pl = None if args.seed is None else args.seed + 1
        seed_rgb_pl = None if args.seed is None else args.seed + 2
        seed_multi_pl = None if args.seed is None else args.seed + 3
        seed_area = None if args.seed is None else args.seed + 4
        
        res_dir = f"{args.output_dir}/{uid}"
        
        # os.makedirs(res_dir, exist_ok=True)
        # json.dump({'scale': scale, 'offset': array2list(offset)}, open(f'{res_dir}/normalize.json', 'w'), indent=4)

        for eye_idx in range(args.num_views):
            # 0. Place Camera, render gt depth map
            view_path = f'{res_dir}/view_{eye_idx}'
            contents = json.load(open(f'{view_path}/cam.json', 'r'))
            fov = contents["fov"]
            c2w =  contents["c2w"]
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            # save cam info
            # json.dump({'c2w': mat2list(c2w), 'fov': fov}, open(f'{view_path}/cam.json', 'w'), indent=4)

            
            for env_map_idx in range(args.num_env_lights):

                env_path = f'{view_path}/env_{env_map_idx}'

                env_contents = json.load(open(f'{env_path}/env.json', 'r'))
                env_map = env_contents["env_map"]
                env_map_path = f'{args.env_map_dir_path}/{env_map}_hdri_2k.exr'
                assert os.path.isfile(env_map_path), env_map
                rotation_euler = env_contents["rotation_euler"]
                strength = env_contents["strength"]
                set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
                
                print("env_map_idx", env_map_idx)
                with stdout_redirected('log_mask.txt'):
                    # render_rgb_and_hint(f'{env_path}/gt')
                    render_rgb(f'{env_path}/gt_obj_mask')
                # # save env map info
                # json.dump({
                #     'env_map': env_map,
                #     'rotation_euler': rotation_euler,
                #     'strength': strength,
                # }, open(f'{env_path}/env.json', 'w'), indent=4)
        # assert False
            


if __name__ == '__main__':
    args: Options = simple_parsing.parse(Options)
    print("options:", args)
    render_core(args)
