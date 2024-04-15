import torch
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

from comfy import model_management
from .attn_handler import SaveAttnInputPatch, InputPatch, ReplacePatch, UnetFunctionWrapper, SamplerCfgFunctionWrapper

from diffusers import UniPCMultistepScheduler

from .oms.garment_diffusion import ClothAdapter
from .oms.OmsDiffusionPipeline import OmsDiffusionPipeline
class AttnStoredExtra:
    def __init__(self,extra) -> None:
        if isinstance(extra, torch.Tensor):
            self.pt = extra.unsqueeze(0)
            self.extra = None
        else:
            self.pt = None
            self.extra = extra
    
    def can_concat(self,other):
        return True
    
    def concat(self, extras):
        if self.pt is not None:  
            out = [self.pt]
            for x in extras:
                out.append(x.pt)
            return torch.cat(out)
        else:
            if self.extra is None:
                return self.extra
            else:
                for x in extras:
                    if x.extra is not None:
                        return x.extra
                return None

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

    CATEGORY = "model_patches"

    def add_features(self, model, clip, seed, sampler_name, scheduler, feature_image, feature_unet_name, enable_feature_guidance=True, feature_guidance_scale=2.5):
        attn_stored_data = self.calculate_features_ldm(clip, seed, sampler_name, scheduler, feature_unet_name, feature_image)
        attn_stored = {}
        attn_stored["enable_feature_guidance"] = enable_feature_guidance
        attn_stored["feature_guidance_scale"] = feature_guidance_scale
        attn_stored["data"] = attn_stored_data
        model = model.clone()
        model.set_model_unet_function_wrapper(UnetFunctionWrapper())
        model.set_model_sampler_cfg_function(SamplerCfgFunctionWrapper())
        model.set_model_attn1_patch(InputPatch())
        for block_name in attn_stored_data.keys():
            for block_number in attn_stored_data[block_name].keys():
                for attention_index in attn_stored_data[block_name][block_number].keys():
                    model.set_model_attn1_replace(
                        ReplacePatch(), block_name, block_number, attention_index)
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
            control = result[4]
            conditioning["c_attn_stored_mult"] = AttnStoredExtra(mult)
            conditioning["c_attn_stored_area"] = AttnStoredExtra(torch.tensor([area[0],area[1],area[2],area[3]]))
            conditioning["c_attn_stored_control"] = AttnStoredExtra(control)
            return result
        comfy.samplers.get_area_and_mult = get_area_and_mult

    def calculate_features_ldm(self, source_clip, seed, sampler_name, scheduler, feature_unet_name, feature_image):
        offload_device = model_management.unet_offload_device()

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
        pipe = OmsDiffusionPipeline.from_pretrained(pipe_path,vae=None,text_encoder=None,torch_dtype=torch.float16)
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
