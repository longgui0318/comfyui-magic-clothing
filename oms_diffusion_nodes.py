import torch
import copy
import inspect
import logging
import uuid
import folder_paths

import comfy.ops
import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.supported_models_base
import comfy.taesd.taesd
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.utils
import comfy.sample
import comfy.samplers

from .attn_patch import InputPatch, OutputPatch


def text_encoder(clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


class ExtractFeaturesWithUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "feature_image": ("LATENT", ),
                 }
                }
    RETURN_TYPES = ("ATTN_STORED",)
    FUNCTION = "extract_features"

    CATEGORY = "loaders"

    def extract_features(self, model, feature_image): 
        latent_image = feature_image["samples"]
        dtype = torch.int32 if model.load_device.type == 'mps' else torch.int64
        timesteps = torch.tensor([0], dtype=dtype, device=model.load_device)
        transformer_options = model.model_options["transformer_options"]
        transformer_options["attn_stored"] = {}
        model.set_model_attn1_patch(InputPatch("save")) 
        _ = model.model.diffusion_model(latent_image, timesteps, transformer_options=transformer_options,).float()
        return (transformer_options["attn_stored"],)
        

class AdditionalFeaturesWithAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "attn_stored": ("ATTN_STORED", ),
                 }
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "add_features"

    CATEGORY = "loaders"
    

    def add_features(self, model,attn_stored):
        transformer_options = model.model_options["transformer_options"]
        transformer_options["do_classifier_free_guidance"] = False
        transformer_options["enable_cloth_guidance"] = False
        transformer_options["attn_stored"] = attn_stored 
        model.set_model_attn1_patch(InputPatch("restore")) 
        model.set_model_attn1_output_patch(OutputPatch("restore"))     
        return model



NODE_CLASS_MAPPINGS = {
    "Extract Features With Unet": ExtractFeaturesWithUnet,
    "Additional Features With Attention": AdditionalFeaturesWithAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract Features With Unet": "Extract Features With Unet",
    "Additional Features With Attention": "Additional Features With Attention",
}
