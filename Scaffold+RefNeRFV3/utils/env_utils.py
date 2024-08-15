import os
import numpy as np
import torch
import nvdiffrast.torch as dr
def cubemap_to_latlong(cubemap, res):
    '''
    cubemap [1,6,size, size, 3]
    res [H, W]
    return img (H, W, 3)
    '''
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap, reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]