import os
import torch
import torch.nn.functional as F
import folder_paths
import numpy as np
from PIL import Image

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

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
from .utils import handle_block_info, clean_attn_stored_memory
from .attn_handler import SaveAttnInputPatch, InputPatch, ReplacePatch, UnetFunctionWrapper, SamplerCfgFunctionWrapper
if hasattr(F, "scaled_dot_product_attention"):
    from .attn_handler import REFAttnProcessor2_0 as REFAttnProcessor
else:
    from .attn_handler import REFAttnProcessor as REFAttnProcessor

#x
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
import argparse

from .oms.garment_diffusion import ClothAdapter
from .oms.OmsDiffusionPipeline import OmsDiffusionPipeline
#x
class AttnStoredExtra:
    def __init__(self,pt) -> None:
        self.pt = pt.unsqueeze(0)
    
    def can_concat(self,other):
        return True
    
    def concat(self, pts):
        out = [self.pt]
        for x in pts:
            out.append(x.pt)
        return torch.cat(out)

class AdditionalFeaturesWithAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "clip": ("CLIP", ),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "feature_image": ("LATENT", ),
                 "feature_unet_name": (folder_paths.get_filename_list("unet"), ),
                 "enable_feature_guidance": ("BOOLEAN", {"default": True}),
                 "feature_guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                 }
                }
    RETURN_TYPES = ("MODEL", "INT", comfy.samplers.KSampler.SAMPLERS,
                    comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("MODEL", "SEED", "SAMPLER_NAME", "SCHEDULER")
    FUNCTION = "add_features"

    CATEGORY = "loaders"

    def add_features(self, model, clip, seed, sampler_name, scheduler, feature_image, feature_unet_name, enable_feature_guidance=True, feature_guidance_scale=2.5):
        attn_stored_data = self.calculate_features_ldm(clip, seed, sampler_name, scheduler, feature_unet_name, feature_image)
        # attn_stored_data = self.calculate_features(model,clip,feature_unet_name, feature_image)
        attn_stored = {}
        attn_stored["enable_feature_guidance"] = enable_feature_guidance
        attn_stored["feature_guidance_scale"] = feature_guidance_scale
        attn_stored["data"] = attn_stored_data
        model = model.clone()
        model.set_model_unet_function_wrapper(UnetFunctionWrapper())
        model.set_model_sampler_cfg_function(SamplerCfgFunctionWrapper())
        model.set_model_attn1_patch(InputPatch())
        # model.set_model_attn2_patch(InputPatch())
        for block_name in attn_stored_data.keys():
            for block_number in attn_stored_data[block_name].keys():
                for attention_index in attn_stored_data[block_name][block_number].keys():
                    model.set_model_attn1_replace(
                        ReplacePatch(), block_name, block_number, attention_index)
                    # model.set_model_attn2_replace(
                    #     ReplacePatch(), block_name, block_number, attention_index)
        self.inject_comfyui()
        model.model_options["transformer_options"]["attn_stored"] = attn_stored
        return (model, seed, sampler_name, scheduler,)

    def inject_comfyui(self):
        old_get_area_and_mult = comfy.samplers.get_area_and_mult
        def get_area_and_mult(self, *args, **kwargs):
            result = old_get_area_and_mult(self, *args, **kwargs)           
            mult = result[1]
            conditioning = result[2]
            area = result[3]
            conditioning["c_attn_stored_mult"] = AttnStoredExtra(mult)
            conditioning["c_attn_stored_area"] = AttnStoredExtra(torch.tensor([area[0],area[1],area[2],area[3]]))
            return result
        comfy.samplers.get_area_and_mult = get_area_and_mult

    def calculate_features(self, source_model, source_clip, feature_unet_name, feature_image):
        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        unet_path = folder_paths.get_full_path("unet", feature_unet_name)
        unet_file = comfy.utils.load_torch_file(unet_path, device=load_device)
        detection_unet_config = model_detection.model_config_from_diffusers_unet(
            unet_file)
        detection_unet_diffusers_keys = comfy.utils.unet_to_diffusers(
            detection_unet_config.unet_config)
        dtype = source_model.model.manual_cast_dtype if source_model.model.manual_cast_dtype is not None else source_model.model.get_dtype()
        workspace_path = os.path.join(os.path.dirname(__file__))
        config_dict = UNet2DConditionModel._dict_from_json_file(
            os.path.join(workspace_path, "unet_config/default_sd15.json"))
        ref_unet = UNet2DConditionModel.from_config(
            config_dict, torch_dtype=dtype)
        ref_unet = ref_unet.to(load_device)
        ref_unet.load_state_dict(unet_file, strict=False)

        attn_store = {}
        attn_proces = {}
        for name in ref_unet.attn_processors.keys():
            if "attn1" in name and "motion_modules" not in name:
                block_name, block_number, attention_index = handle_block_info(
                    name, detection_unet_diffusers_keys)
            else:
                block_name = None
                block_number = 0
                attention_index = 0
            if block_name is not None:
                attn_proces[name] = REFAttnProcessor(
                    True, block_name, block_number, attention_index)
            else:
                attn_proces[name] = REFAttnProcessor(False)
        ref_unet.set_attn_processor(attn_proces)
        latent_image = feature_image["samples"]
        latent_image = latent_image.to(load_device)
        tokens = source_clip.tokenize("")
        prompt_embeds_null = source_clip.encode_from_tokens(
            tokens, return_pooled=False)
        prompt_embeds_null = prompt_embeds_null.to(load_device)
        ref_unet(latent_image, 0, prompt_embeds_null,
                 cross_attention_kwargs={"attn_store": attn_store})
        del ref_unet
        del prompt_embeds_null
        latent_image = latent_image.to(offload_device)
        return attn_store

    def calculate_features_ldm(self, source_clip, seed, sampler_name, scheduler, feature_unet_name, feature_image):
        
        offload_device = model_management.unet_offload_device()
        load_device = model_management.get_torch_device()

        unet_path = folder_paths.get_full_path("unet", feature_unet_name)
        model_patcher = comfy.sd.load_unet(unet_path)
        model_patcher.set_model_attn1_patch(SaveAttnInputPatch())
        attn_stored = {}
        attn_stored["data"] = {}
        model_patcher.model_options["transformer_options"]["attn_stored"] = attn_stored

        latent_image = feature_image["samples"]
        if latent_image.shape[0] > 1:
            latent_image = torch.chunk(latent_image, latent_image.shape[0])[0]
        noise = comfy.sample.prepare_noise(latent_image, seed, None)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        positive_tokens = source_clip.tokenize("")
        positive_cond, positive_pooled = source_clip.encode_from_tokens(
            positive_tokens, return_pooled=True)
        positive = [[positive_cond, {"pooled_output": positive_pooled}]]
        negative = []
        samples = comfy.sample.sample(model_patcher, noise, 1, 1, sampler_name, scheduler,
                                      positive, negative, latent_image, denoise=1.0,
                                      disable_noise=False, start_step=None,
                                      last_step=None, force_full_denoise=False,
                                      noise_mask=None, callback=None, disable_pbar=disable_pbar, seed=seed)
        del positive_cond
        del positive_pooled
        del positive_tokens
        del model_patcher
        del samples

        latent_image = feature_image["samples"].to(offload_device)
        return attn_stored["data"]

class RunOmsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cloth_latent": ("LATENT",),
                             "gen_latent": ("LATENT",),
                             "model": ("MODEL",),
                             "clip": ("CLIP", ),
                             "positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 10.0,"step": 0.01}),
                             "cloth_guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0,"step": 0.01}),
                             "steps": ("INT", {"default": 25, "min": 0, "max": 100}),
                             "height": ("INT", {"default": 768, "min": 0, "max": 2048}),
                             "width": ("INT", {"default": 512, "min": 0, "max": 2048}),
                             }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run_oms"

    CATEGORY = "loaders"
    
    def run_oms(self,cloth_latent,gen_latent, model,clip,positive,negative,seed,scale,cloth_guidance_scale,steps,height,width):
        tokens = clip.tokenize("")
        cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
        
        num_samples = 1
        prompt_embeds_null = cond
        gen_latent["samples"] = model.generate(cloth_latent["samples"],gen_latent["samples"], prompt_embeds_null, positive[0][0], negative[0][0], num_samples, seed, scale, cloth_guidance_scale, steps, height, width)
        return (gen_latent,)

class LoadOmsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "run_oms"

    CATEGORY = "loaders"
    
    def run_oms(self, seed):
        unet_path = folder_paths.get_full_path("unet", "oms_diffusion_768_200000.safetensors")
        pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        pipe = OmsDiffusionPipeline.from_pretrained(pipe_path,torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        del pipe.vae
        del pipe.text_encoder
        full_net = ClothAdapter(pipe, unet_path, "cuda",True)
        return (full_net,)

NODE_CLASS_MAPPINGS = {
    "Additional Features With Attention": AdditionalFeaturesWithAttention,
    "RUN OMS": RunOmsNode,
    "LOAD OMS": LoadOmsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Additional Features With Attention": "Additional Features With Attention",
    "RUN OMS": "RUN OMS",
    "LOAD OMS": "LOAD OMS",
}
