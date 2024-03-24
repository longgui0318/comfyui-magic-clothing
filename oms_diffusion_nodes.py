import torch
import copy
import inspect
import logging
import uuid
import folder_paths

import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.supported_models_base
import comfy.taesd.taesd

class ExtractFeaturesWithAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "clip": ("CLIP", ),
                 "feature_image": ("LATENT", ),
                 }
                }
    RETURN_TYPES = ("MODEL", "LATENT",)
    FUNCTION = "unet_extract_features"

    CATEGORY = "loaders"

    def unet_extract_features(self, model,clip, feature_image):
        

        tokens = clip.tokenize("")
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        print(id(model))
        model = model.clone()
        print(id(model))
        if model.model.diffusion_model.in_channels ==9:
            # 如果输入的通道数是9，转成4，这时候，需要注意的就是要重新构建一个出来，不然这个数据会出现错误
            print(id(model.model))
            model.model =copy.copy(model.model)
            print(id(model.model))
            model.model.diffusion_model = UNetModel()
            
        # if model.model.in_channels
        # self.inner_model, x, timestep, uncond, cond, cond_scale, model_options=model_options, seed=seed

        
        return model, feature_image


NODE_CLASS_MAPPINGS = {
    "Extract Features With Attention": ExtractFeaturesWithAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract Features With Attention": "Extract Features With Attention",
}
