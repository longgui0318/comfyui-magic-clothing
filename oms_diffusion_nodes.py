import os
import torch
import torch.nn.functional as F
import copy
import inspect
import logging
import uuid
import folder_paths

import comfy.ops
import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.model_base
import comfy.supported_models_base
import comfy.taesd.taesd
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.utils
import comfy.sample
import comfy.samplers

from diffusers import UNet2DConditionModel
from comfy import model_detection
from comfy import model_management
from .utils import handle_block_info
from .attn_patch import InputPatch, ReplacePatch
if hasattr(F, "scaled_dot_product_attention"):
    from .attn_handler import REFAttnProcessor2_0 as REFAttnProcessor
else:
    from .attn_handler import REFAttnProcessor as REFAttnProcessor


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
        _ = model.model.diffusion_model(
            latent_image, timesteps, transformer_options=transformer_options,).float()
        return (transformer_options["attn_stored"],)


class AdditionalFeaturesWithAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "clip": ("CLIP", ),
                 "feature_unet_name": (folder_paths.get_filename_list("unet"), ),
                 "feature_image": ("LATENT", ),
                 }
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "add_features"

    CATEGORY = "loaders"

    def calculate_features(self, source_model, source_clip, feature_unet_name, feature_image):
        state_dict = source_model.model.diffusion_model.state_dict()
        for _key in state_dict.keys():
            print("source_model",_key,state_dict[_key].shape)
        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        unet_path = folder_paths.get_full_path("unet", feature_unet_name)
        unet_file = comfy.utils.load_torch_file(unet_path,device=load_device)
        dtype = source_model.model.manual_cast_dtype if source_model.model.manual_cast_dtype is not None else source_model.model.get_dtype()
        workspace_path = os.path.join(os.path.dirname(__file__))
        config_dict = UNet2DConditionModel._dict_from_json_file(os.path.join(workspace_path, "unet_config/default_sd15.json"))
        ref_unet = UNet2DConditionModel.from_config(config_dict, torch_dtype=dtype)
        ref_unet = ref_unet.to(load_device)
        ref_unet.load_state_dict(unet_file, strict=False)
        
        state_dict = ref_unet.state_dict()
        for _key in state_dict.keys():
            print("ref_unet",_key,state_dict[_key].shape)
        
        attn_store = {}
        attn_proces = {}
        for name in ref_unet.attn_processors.keys():
            if "attn1" in name and "motion_modules" not in name:
                block_name,block_number,attention_index =handle_block_info(name)
                attn_proces[name] = REFAttnProcessor(True,block_name,block_number,attention_index)
            else:
                attn_proces[name] = REFAttnProcessor(False)
        ref_unet.set_attn_processor(attn_proces)
        latent_image = feature_image["samples"]
        latent_image = latent_image.to(load_device)
        tokens = source_clip.tokenize("")
        prompt_embeds_null = source_clip.encode_from_tokens(tokens, return_pooled=False)
        prompt_embeds_null = prompt_embeds_null.to(load_device)
        ref_unet(latent_image, 0, prompt_embeds_null,cross_attention_kwargs={"attn_store": attn_store})
        del ref_unet
        del prompt_embeds_null
        latent_image = latent_image.to(offload_device)
        return attn_store
    
    def add_features(self, model, clip, feature_unet_name, feature_image):
        attn_stored = self.calculate_features(model, clip, feature_unet_name, feature_image)
        transformer_options = model.model_options["transformer_options"]
        transformer_options["do_classifier_free_guidance"] = False
        transformer_options["enable_cloth_guidance"] = False
        transformer_options["attn_stored"] = attn_stored
        model = model.clone()
        model.set_model_attn1_patch(InputPatch())
        for block_name in attn_stored.keys():
            for block_number  in attn_stored[block_name].keys():
                for attention_index in attn_stored[block_name][block_number].keys():
                    model.set_model_attn1_replace(ReplacePatch(),block_name,block_number,attention_index)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "Extract Features With Unet": ExtractFeaturesWithUnet,
    "Additional Features With Attention": AdditionalFeaturesWithAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract Features With Unet": "Extract Features With Unet",
    "Additional Features With Attention": "Additional Features With Attention",
}
