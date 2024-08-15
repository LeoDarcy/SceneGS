import os
import numpy as np
from bvh import RayTracer
import torch

points_n = 10
means3D = torch.rand((points_n,3), device="cuda")
get_scaling = torch.rand((points_n, 3), device="cuda")
get_rotation = torch.rand((points_n, 6), device="cuda")

raytracer = RayTracer(means3D, get_scaling, get_rotation)

search_n = 3
normal = torch.rand((search_n,3), device="cuda")
rand_rays_o = torch.rand((search_n, 3), device="cuda")
rand_rays_d = torch.rand((search_n, 3), device="cuda")
cov_inv = torch.rand((search_n, 6), device="cuda")
opacity =  torch.rand((points_n, 6), device="cuda")
trace_results = raytracer.trace_visibility(rand_rays_o, rand_rays_d, means3D, cov_inv, opacity, normal)
print(trace_results)