import numpy as np
import cv2
import imageio
import os
def load_img(target, zero_one=True):
    assert os.path.isfile(target), target
    if target[-4:] == '.png' or target[-4:] == '.jpg' :
        target = cv2.imread(target, -1)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        if zero_one:
            target = target / 255.  # Normalize target images to [-1, 1].
        else:
            target = target / 127.5 - 1.  # Normalize target images to [-1, 1].
    elif target[-4:] == '.exr':
        # target = imageio.v3.imread(target)
        target = cv2.imread(target, -1)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    else:
        assert False, target
    return target
def get_mask(img):
    # get obj mask and plane mask
    H, W, C = img.shape
    if C == 4:
        alpha_mask = img[:, :, 3:4]
        img = img[:, :,:3] * alpha_mask 
    else:
        alpha_mask = np.ones((H, W, 1))
    # 提取 R、G、B 通道
    b_channel = img[:, :, 0:1]
    g_channel = img[:, :, 1:2]
    r_channel = img[:, :, 2:3]

    # 根据条件进行更改& (g_channel == 0) & (b_channel == 0) & (r_channel == 0) & (b_channel == 0) 
    condition1 = (r_channel > 200)  & (alpha_mask >0)  #Obj
    condition2 = (g_channel > 128) & (alpha_mask >0) # Plane

    # cv2.imwrite("obj.png", ((condition1 )).astype(np.uint8) *255 )
    # cv2.imwrite("plane.png", ((condition2 )).astype(np.uint8) *255 )
    # cv2.imwrite("input.png", ((img )).astype(np.uint8))
    # cv2.imwrite("or.png", ((condition1 | condition2) == (alpha_mask >0)).astype(np.uint8) *255 )
    # cv2.imwrite("and.png", ((condition1 & condition2)).astype(np.uint8) *255 )
    condition1 = (~condition2) & (alpha_mask >0)
    assert np.any(condition1 & condition2) == False #没有同时为obj和plane的mask
    assert np.all((condition1 | condition2) & (alpha_mask >0)) == True # obj和plane merge 后和alpha一致
    return condition1, condition2