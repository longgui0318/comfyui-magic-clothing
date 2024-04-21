import copy
import torch
import folder_paths
import os
import numpy as np

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

from PIL import Image
from .utils import pt_hash
from comfy import model_management
from .attn_handler import SaveAttnInputPatch, InputPatch, ReplacePatch, UnetFunctionWrapper, SamplerCfgFunctionWrapper

from diffusers import UniPCMultistepScheduler,AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline

from .oms.garment_diffusion import ClothAdapter
from .oms.OmsDiffusionPipeline import OmsDiffusionPipeline
from .oms.utils import prepare_image
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
    
class InjectTensorHashLog:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"enable": ("BOOLEAN", {"default": True}),
                 }
                }
    RETURN_TYPES = ("INT",)
    FUNCTION = "inject"

    CATEGORY = "model_patches"

    def inject(self,enable):
        torch.Tensor.__hash_log__ = pt_hash
        return (0,)
    
class ChangePixelValueNormalization:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                { "pixels": ("IMAGE", ),
                  "mode": (["[0,1]=>[-1,1]", "[-1,1]=>[0,1]"],),            
                 }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalization"

    CATEGORY = "image"

    def normalization(self,pixels,mode):
        if mode == "[0,1]=>[-1,1]":
            pixels = (pixels * 255).round().clamp(min=0,max=255) / 127.5 - 1.0
        elif mode == "[-1,1]=>[0,1]":
            pixels = ((pixels+1) * 127.5).clamp(min=0,max=255) / 255.0
        else:
            pixels = pixels
        return (pixels,)

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
        torch.Tensor.__hash_log__ = pt_hash
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
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        positive_tokens = source_clip.tokenize("")
        positive_cond, positive_pooled = source_clip.encode_from_tokens(
            positive_tokens, return_pooled=True)
        positive = [[positive_cond, {"pooled_output": positive_pooled}]]
        negative = []
        # sigmas = comfy.samplers.calculate_sigmas(magicClothingModel.model.model_sampling,scheduler,1).to(magicClothingModel.load_device)
        
        dtype = magicClothingModel.model.get_dtype()
        # if not os.path.exists(folder_paths.get_output_directory()):
        #     os.makedirs(folder_paths.get_output_directory())
        # timestep = sigmas[0].expand((latent_image.shape[0])).to(dtype)
        latent_image = latent_image.to(magicClothingModel.load_device).to(dtype)
        noise = noise.to(magicClothingModel.load_device).to(dtype)  
        # context = positive_cond.to(magicClothingModel.load_device).to(dtype)    
        # model_management.load_model_gpu(magicClothingModel)                      
        # magicClothingModel.model.diffusion_model(latent_image, timestep, context=context, control=None, transformer_options=magicClothingModel.model_options["transformer_options"])
        sigmas = torch.tensor([1,0])
        samples = comfy.sample.sample(magicClothingModel, noise, 1, 1, sampler_name, scheduler,
                                      positive, negative, latent_image, denoise=1.0,
                                      disable_noise=False, start_step=None,
                                      last_step=None, force_full_denoise=False,sigmas=sigmas,
                                      noise_mask=None, callback=None, disable_pbar=disable_pbar, seed=41)
        del positive_cond
        del positive_pooled
        del positive_tokens
        latent_image = feature_image["samples"].to(model_management.unet_offload_device())
        return attn_stored["data"]

class RunOmsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cloth_image": ("IMAGE",),
                             "model": ("MODEL",),
                            #  "clip": ("CLIP", ),
                            #  "positive": ("CONDITIONING", ),
                            #  "negative": ("CONDITIONING", ),
                             "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                             "scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 10.0,"step": 0.01}),
                             "cloth_guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0,"step": 0.01}),
                             "steps": ("INT", {"default": 25, "min": 0, "max": 100}),
                             "height": ("INT", {"default": 768, "min": 0, "max": 2048}),
                             "width": ("INT", {"default": 576, "min": 0, "max": 2048}),
                             }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_oms"

    CATEGORY = "loaders"
    
    def run_oms(self,cloth_image, model,seed,scale,cloth_guidance_scale,steps,height,width):
        # seed = 1234
        # width = 96
        # height = 96
        # steps = 1
        # cloth_image =  (cloth_image * 255).round().clamp(min=0,max=255).to(dtype=torch.float32)  / 255.0
        cloth_image =  (cloth_image * 255).round().clamp(min=0,max=255).to(dtype=torch.float32)  / 127.5 - 1.0
        cloth_image =  cloth_image.permute(0,3,1,2)
        if not isinstance(model,ClothAdapter):
            gen_image = cloth_image.permute(0, 2, 3, 1)
            gen_image = ((gen_image+1) * 127.5).clamp(min=0,max=255).to(dtype=torch.float32) / 255.0
            return (gen_image,)
        cloth_image =  cloth_image.to(model.device, dtype=model.pipe.dtype)
        cloth_image.__hash_log__("特征提取-衣服")
        with torch.inference_mode():
            prompt_embeds_null = model.pipe.encode_prompt([""], device=model.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
            prompt_embeds_null.__hash_log__("特征提取-空提示")
            prompt_embeds, negative_prompt_embeds = model.pipe.encode_prompt(
                "a photography of a model,best quality, high quality",
                model.pipe.device,
                1,
                True,
                "bare, monochrome, lowres, bad anatomy, worst quality, low quality",
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            cloth_latent = {}
            cloth_latent["samples"] = model.pipe.vae.encode(cloth_image).latent_dist.mode()
        gen_image = model.generate(cloth_latent["samples"],None, prompt_embeds_null,prompt_embeds, negative_prompt_embeds, 1, seed, scale, cloth_guidance_scale, steps, height, width)
        with torch.inference_mode():
            gen_image = model.pipe.vae.decode(gen_image, return_dict=False, generator=model.generator)[0]
        gen_image.__hash_log__("生成-图像(完)")
        gen_image = gen_image.permute(0, 2, 3, 1)
        # gen_image = (gen_image * 255.0).clamp(min=0,max=255).to(dtype=torch.float32) / 255.0

        gen_image = ((gen_image+1) * 127.5).clamp(min=0,max=255).to(dtype=torch.float32) / 255.0
        return (gen_image,)

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
        torch.Tensor.__hash_log__ = pt_hash
        if "768" in magicClothingUnet:
            return ({},)
        unet_path = folder_paths.get_full_path("unet", magicClothingUnet)
        unet_path = "C:\\Users\\Administrator\\.cache\\models\\oms_diffusion_768_200000.safetensors"
        pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
        pipe = OmsDiffusionPipeline.from_pretrained(pipe_path,vae=vae,torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        full_net = ClothAdapter(pipe, unet_path, "cuda",True)
        return (full_net,)

NODE_CLASS_MAPPINGS = {
    "InjectTensorHashLog":InjectTensorHashLog,
    "VAE Mode Choose": VAEModeChoose,
    "Change Pixel Value Normalization":ChangePixelValueNormalization,
    "Load Magic Clothing Model": LoadMagicClothingModel,
    "Add Magic Clothing Attention": AddMagicClothingAttention,
    "RUN OMS": RunOmsNode,
    "LOAD OMS": LoadOmsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InjectTensorHashLog":"InjectTensorHashLog",
    "VAE Mode Choose":"VAE Mode Choose",
    "Change Pixel Value Normalization":"Change Pixel Value Normalization",
    "Load Magic Clothing Model": "Load Magic Clothing Model",
    "Add Magic Clothing Attention": "Add Magic Clothing Attention",
    "RUN OMS": "RUN OMS",
    "LOAD OMS": "LOAD OMS",
}
