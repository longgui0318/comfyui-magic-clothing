import copy
import torch
import folder_paths

import comfy.ops
import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.model_base
import comfy.model_detection
import comfy.supported_models_base
import comfy.taesd.taesd
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.ldm.models.autoencoder
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

class VAEModeChoose:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"vae": ("VAE",),
                 "mode": (["sample", "mode"],),
                 }
                }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "choose_vae_mode"

    CATEGORY = "model_patches"

    def choose_vae_mode(self, vae, mode):
        if isinstance(vae.first_stage_model.regularization, comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer):
            vae.first_stage_model.regularization.sample = mode == "sample"
        return (vae,)

class LoadMagicClothingModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"sourceModel": ("MODEL",),
                 "magicClothingUnet": (folder_paths.get_filename_list("unet"), ),
                 }
                }
    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("sourceModel", "magicClothingModel")
    FUNCTION = "load_unet"

    CATEGORY = "model_patches"

    def load_unet(self, sourceModel, magicClothingUnet):
        unet_path = folder_paths.get_full_path("unet", magicClothingUnet)
        unet_state_dict = comfy.utils.load_torch_file(unet_path)
        model_config = copy.deepcopy(sourceModel.model.model_config)
        if model_config.unet_config["in_channels"] == 9:
            model_config.unet_config["in_channels"] = 4
            model_config.unet_config["model_channels"] = 320
        
        source_state_dict = sourceModel.model.diffusion_model.state_dict()
        
        diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

        new_sd = {}
        for k in diffusers_keys:
            ldm_k = diffusers_keys[k]
            if k in unet_state_dict:
                new_sd[diffusers_keys[k]] = unet_state_dict.pop(k)
            elif ldm_k in source_state_dict:
                new_sd[ldm_k] = source_state_dict[ldm_k]
        
        parameters = comfy.utils.calculate_parameters(new_sd)
        
        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        model = model_config.get_model(new_sd, "")
        model = model.to(offload_device)
        model.load_model_weights(new_sd, "")
        left_over = unet_state_dict.keys()
        if len(left_over) > 0:
            print("left over keys in unet: {}".format(left_over))
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (sourceModel,model_patcher)


class AddMagicClothingAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"sourceModel": ("MODEL",),
                 "magicClothingModel": ("MODEL",),
                 "clip": ("CLIP", ),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "feature_image": ("LATENT", ),
                 "enable_feature_guidance": ("BOOLEAN", {"default": True}),
                 "feature_guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                 }
                }
    RETURN_TYPES = ("MODEL",comfy.samplers.KSampler.SAMPLERS,comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("MODEL", "SAMPLERS","SCHEDULER")
    FUNCTION = "add_features"

    CATEGORY = "model_patches"

    def add_features(self, sourceModel,magicClothingModel, clip,sampler_name, scheduler, feature_image, enable_feature_guidance=True, feature_guidance_scale=2.5):
        attn_stored_data = self.calculate_features(magicClothingModel,clip,sampler_name, scheduler, feature_image)
        attn_stored = {}
        attn_stored["enable_feature_guidance"] = enable_feature_guidance
        attn_stored["feature_guidance_scale"] = feature_guidance_scale
        attn_stored["data"] = attn_stored_data
        sourceModel = sourceModel.clone()
        sourceModel.set_model_unet_function_wrapper(UnetFunctionWrapper())
        sourceModel.set_model_sampler_cfg_function(SamplerCfgFunctionWrapper())
        sourceModel.set_model_attn1_patch(InputPatch())
        for block_name in attn_stored_data.keys():
            for block_number in attn_stored_data[block_name].keys():
                for attention_index in attn_stored_data[block_name][block_number].keys():
                    sourceModel.set_model_attn1_replace(ReplacePatch(), block_name, block_number, attention_index)
        self.inject_comfyui()
        sourceModel.model_options["transformer_options"]["attn_stored"] = attn_stored
        return (sourceModel, sampler_name,scheduler,)

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

    def calculate_features(self,magicClothingModel, source_clip,sampler_name,scheduler, feature_image):
        magicClothingModel.set_model_attn1_patch(SaveAttnInputPatch())
        attn_stored = {}
        attn_stored["data"] = {}
        magicClothingModel.model_options["transformer_options"]["attn_stored"] = attn_stored

        latent_image = feature_image["samples"]
        if latent_image.shape[0] > 1:
            latent_image = torch.chunk(latent_image, latent_image.shape[0])[0]
        noise = torch.zeros_like(latent_image)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        positive_tokens = source_clip.tokenize("")
        positive_cond, positive_pooled = source_clip.encode_from_tokens(
            positive_tokens, return_pooled=True)
        positive = [[positive_cond, {"pooled_output": positive_pooled}]]
        negative = []
        # sigmas = comfy.samplers.calculate_sigmas(magicClothingModel.model.model_sampling,scheduler,1).to(magicClothingModel.load_device)
        
        dtype = magicClothingModel.model.get_dtype()
        
        # timestep = sigmas[0].expand((latent_image.shape[0])).to(dtype)
        latent_image = latent_image.to(magicClothingModel.load_device).to(dtype)
        noise = noise.to(magicClothingModel.load_device).to(dtype)  
        # context = positive_cond.to(magicClothingModel.load_device).to(dtype)    
        # model_management.load_model_gpu(magicClothingModel)                      
        # magicClothingModel.model.diffusion_model(latent_image, timestep, context=context, control=None, transformer_options=magicClothingModel.model_options["transformer_options"])
        samples = comfy.sample.sample(magicClothingModel, noise, 1, 1, sampler_name, scheduler,
                                      positive, negative, latent_image, denoise=1.0,
                                      disable_noise=False, start_step=None,
                                      last_step=None, force_full_denoise=False,
                                      noise_mask=None, callback=None, disable_pbar=disable_pbar, seed=41)
        del positive_cond
        del positive_pooled
        del positive_tokens
        latent_image = feature_image["samples"].to(model_management.unet_offload_device())
        return attn_stored["data"]

class RunOmsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cloth_latent": ("LATENT",),
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
    
    def run_oms(self,cloth_latent, model,clip,positive,negative,seed,scale,cloth_guidance_scale,steps,height,width):
        tokens = clip.tokenize("")
        cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
        num_samples = 1
        prompt_embeds_null = cond
        cloth_latent["samples"] = model.generate(cloth_latent["samples"],None, prompt_embeds_null,positive[0][0], negative[0][0], num_samples, seed, scale, cloth_guidance_scale, steps, height, width,timesteps=None)
        return (cloth_latent,)

class LoadOmsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"magicClothingUnet": (folder_paths.get_filename_list("unet"), ),
                 }
                }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "run_oms"

    CATEGORY = "loaders"
    
    def run_oms(self, magicClothingUnet):
        unet_path = folder_paths.get_full_path("unet", magicClothingUnet)
        pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        pipe = OmsDiffusionPipeline.from_pretrained(pipe_path,text_encoder=None)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        full_net = ClothAdapter(pipe, unet_path, "cuda",True)
        return (full_net,)

NODE_CLASS_MAPPINGS = {
    "VAE Mode Choose": VAEModeChoose,
    "Load Magic Clothing Model": LoadMagicClothingModel,
    "Add Magic Clothing Attention": AddMagicClothingAttention,
    "RUN OMS": RunOmsNode,
    "LOAD OMS": LoadOmsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAE Mode Choose":"VAE Mode Choose",
    "Load Magic Clothing Model": "Load Magic Clothing Model",
    "Add Magic Clothing Attention": "Add Magic Clothing Attention",
    "RUN OMS": "RUN OMS",
    "LOAD OMS": "LOAD OMS",
}
