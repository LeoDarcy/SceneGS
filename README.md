# 2DGS+XXXX

Code base: 2DGS

Target：enable learnable environment map

Modification: gaussian_renderer/\_\_init\_\_.py


# HashScaffold+ShadingEnvironment

Code base: Scaffold + HashGrid (gaussian_model.py)

Modification: gaussian_renderer/\_\_init\_\_.py Line 394 

train.py Line49: BJY_RenderingSettings ={"env_color" , "bounce"} 

"env_color": enable learnable environment map

"bounce": enable reflection from other Gaussians
