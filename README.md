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


# HashScaffold+RefNeRF

Code base: Scaffold + HashGrid (gaussian_renderer/\_\_init\_\_.py Line 27=Scaffold, Line 28= HashGrid)

Using Integrated Directional Encoding in RefNeRF (gaussian_model.py Line 241)

Using 8-layer MLP as specular decoder (gaussian_model.py Line 155)

Option: Using 2-layer MLP as specular decoder (gaussian_model.py Line 149-154)

Loss: Color loss + normal loss (predicted_normal_loss)




