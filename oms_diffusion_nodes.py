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
        model = model.clone()
        unet = model.model
        if model.model.in_channels()
        unet.
        self.inner_model, x, timestep, uncond, cond, cond_scale, model_options=model_options, seed=seed

        
        return model, feature_image


NODE_CLASS_MAPPINGS = {
    "Extract Features With Attention": ExtractFeaturesWithAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract Features With Attention": "Extract Features With Attention",
}
