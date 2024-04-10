import os
import torch
import torch.nn.functional as F
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
from .utils import handle_block_info, clean_attn_stored_memory
from .attn_handler import SaveAttnInputPatch, InputPatch, ReplacePatch, UnetFunctionWrapper, SamplerCfgFunctionWrapper
if hasattr(F, "scaled_dot_product_attention"):
    from .attn_handler import REFAttnProcessor2_0 as REFAttnProcessor
else:
    from .attn_handler import REFAttnProcessor as REFAttnProcessor

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
        attn_stored_data = self.calculate_features_ldm(
            clip, seed, sampler_name, scheduler, feature_unet_name, feature_image)
        attn_stored = {}
        attn_stored["enable_feature_guidance"] = enable_feature_guidance
        attn_stored["feature_guidance_scale"] = feature_guidance_scale
        attn_stored["data"] = attn_stored_data
        model = model.clone()
        model.set_model_unet_function_wrapper(UnetFunctionWrapper())
        model.set_model_sampler_cfg_function(SamplerCfgFunctionWrapper())
        model.set_model_attn1_patch(InputPatch())
        model.set_model_attn2_patch(InputPatch())
        for block_name in attn_stored_data.keys():
            for block_number in attn_stored_data[block_name].keys():
                for attention_index in attn_stored_data[block_name][block_number].keys():
                    model.set_model_attn1_replace(
                        ReplacePatch(), block_name, block_number, attention_index)
                    model.set_model_attn2_replace(
                        ReplacePatch(), block_name, block_number, attention_index)
        self.inject_comfyui(attn_stored)
        model.model_options["transformer_options"]["attn_stored"] = attn_stored
        return (model, seed, sampler_name, scheduler,)

    def inject_comfyui(self, attn_stored_ref):
        old_get_area_and_mult = comfy.samplers.get_area_and_mult

        def get_area_and_mult(self, *args, **kwargs):
            result = old_get_area_and_mult(self, *args, **kwargs)
            mult = result[1]
            area = result[3]
            input_x = result[0]
            if attn_stored_ref is not None:
                check_key = ["cond_or_uncond_out_cond", "cond_or_uncond_out_uncond", "out_cond_init",
                             "out_count_init", "cond_or_uncond_extra_options", "cond_or_uncond_replenishment"]
                need_clean_memory = False
                for key in check_key:
                    if key in attn_stored_ref:
                        need_clean_memory = True
                if need_clean_memory:
                    clean_attn_stored_memory(attn_stored_ref)
                if "input_x_extra_options" not in attn_stored_ref:
                    attn_stored_ref["input_x_extra_options"] = []
                attn_stored_ref["input_x_extra_options"].append({
                    "input_x": input_x,
                    "mult": mult,
                    "area": area
                })
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
        model_patcher.set_model_attn2_patch(SaveAttnInputPatch())
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


NODE_CLASS_MAPPINGS = {
    "Additional Features With Attention": AdditionalFeaturesWithAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Additional Features With Attention": "Additional Features With Attention",
}
