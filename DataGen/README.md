# 第一步
先做好json，后续直接从json中读取就可以 FOV之前是25-35 现在设置为85-95
python 

# 第二步：做GT RGB和物体Mask
渲染两张图，一张有物体和一张没有物体的

(可能可以渲染相同位置视角下一张均匀光照下只有物体的图，用于作为输入物体的原RGB)

渲染物体和平板的mask

# 第三步： 做GT shadow的mask
基于mask以及两张有无物体的图，做GT mask

render_gtShadow_step3.py

# 第四步： 做镜面效果和AO，设置物体的镜面为镜面
render_AO_step4.py

render_Refl_step4.py


# 第五步： 训练


# 文件结构：


Data
    
    --HDRI
    --assets
        --objaverse_fids
        polyhaven_hdris_origin.json
    --Output

        --物体id
            --view{0-3}
                cam.json
                --env_{0-3}
                    env.json
                    gt.png#完整插入
                    gt_noObj.png#没有物体
                    gt_shadow.png# 阴影的mask wight
                    gt_obj_mask.png# 物体的mask，平板的mask
                    gt_refl.png #反射镜面
                    gt_ao.png # ao效果 
                --area_light#均匀光照下的obj