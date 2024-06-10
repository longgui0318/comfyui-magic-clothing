import copy
import torch
import folder_paths

import comfy.model_patcher
import comfy.ldm.models.autoencoder
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.sampler_helpers

from .utils import pt_hash
from comfy import model_management
from .attn_handler import SaveAttnInputPatch, InputPatch, ReplacePatch, UnetFunctionWrapper, SamplerCfgFunctionWrapper

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

    CATEGORY = "loaders"

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
                 "enable_feature_guidance": ("BOOLEAN", {"default": True}),
                 "feature_image": ("LATENT", ),
                 "feature_guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                #  "sigma": ("FLOAT", {"default": 0.71, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.01}),
                #  "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                #  "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                #  "sigma": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step": 0.05}),
                #  "start_step":("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                #  "end_step":("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                #  "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                 }
                }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)

    FUNCTION = "add_features"

    CATEGORY = "model_patches"

    def add_features(self, sourceModel,magicClothingModel, clip,enable_feature_guidance ,feature_image,feature_guidance_scale,
                    #  sigma,sampler_name,scheduler,start_step=0,end_step = 100,steps = 20,
                     ):
        attn_stored = self.calculate_features_zj(magicClothingModel,clip, feature_image)
        attn_stored["enable_feature_guidance"] = enable_feature_guidance
        attn_stored["feature_guidance_scale"] = feature_guidance_scale
        attn_stored_data = attn_stored["data"]
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
        return (sourceModel,)

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

    def calculate_features(self,magicClothingModel, source_clip,feature_image,sigma =None,start_step =None,end_step =None,steps =None,scheduler =None,sampler_name =None):
        magicClothingModel.set_model_attn1_patch(SaveAttnInputPatch())
        attn_stored = {}
        attn_stored["data"] = {}
        magicClothingModel.model_options["transformer_options"]["attn_stored"] = attn_stored

        latent_image = feature_image["samples"]
        if latent_image.shape[0] > 1:
            latent_image = torch.chunk(latent_image, latent_image.shape[0])[0]
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise = noise+0
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        positive_tokens = source_clip.tokenize("")
        positive_cond, positive_pooled = source_clip.encode_from_tokens(
            positive_tokens, return_pooled=True)
        positive = [[positive_cond, {"pooled_output": positive_pooled}]]
        negative = []        
        dtype = magicClothingModel.model.get_dtype()
        latent_image = latent_image.to(magicClothingModel.load_device).to(dtype)
        noise = noise.to(magicClothingModel.load_device).to(dtype)  
        sigmas = torch.tensor([1,0])
        samples = comfy.sample.sample(magicClothingModel, noise, 1, 1, "uni_pc", "karras",
                                      positive, negative, latent_image, denoise=1.0,
                                      disable_noise=False, start_step=None,
                                      last_step=None, force_full_denoise=False,sigmas=sigmas,
                                      noise_mask=None, callback=None, disable_pbar=disable_pbar, seed=41)
        del positive_cond
        del positive_pooled
        del positive_tokens
        latent_image = feature_image["samples"].to(model_management.unet_offload_device())
        return attn_stored
    
    def _calculate_sigmas(self,steps,model_sampling,scheduler,sampler_name):
        sigmas = None

        discard_penultimate_sigma = False
        if sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = comfy.samplers.calculate_sigmas(model_sampling,scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
        
    def calculate_features_zj(self,magicClothingModel, source_clip,feature_image,sigma = 0,start_step =None,end_step =None,steps =None,scheduler =None,sampler_name =None):
        magicClothingModel.set_model_attn1_patch(SaveAttnInputPatch())
        attn_stored = {}
        attn_stored["data"] = {}
        magicClothingModel.model_options["transformer_options"]["attn_stored"] = attn_stored

        latent_image = feature_image["samples"]
        if latent_image.shape[0] > 1:
            latent_image = torch.chunk(latent_image, latent_image.shape[0])[0]
        positive_tokens = source_clip.tokenize("")
        positive_cond, positive_pooled = source_clip.encode_from_tokens(positive_tokens, return_pooled=True)
        dtype = magicClothingModel.model.get_dtype()
        
        latent_image = magicClothingModel.model.process_latent_in(latent_image).to(magicClothingModel.load_device)
        context = positive_cond.to(magicClothingModel.load_device).to(dtype)
        # sigmas = self._calculate_sigmas(steps,magicClothingModel.model.model_sampling,scheduler,sampler_name)
        # sigmas = sigmas.to(magicClothingModel.load_device)
        # start_step = max(0, min(start_step, steps))
        # end_step = max(0, min(end_step, steps))
        # calc_steps = sigmas[start_step:end_step]
        # calc_sigmas = [calc_steps[i].item() for i in range(calc_steps.shape[0])]
        # attn_stored["calc_sigmas"] = calc_sigmas
        # real_sigma = sigmas[0].expand((latent_image.shape[0]))
        # real_sigma = (real_sigma*0+sigma).to(dtype)
        real_sigma = torch.tensor([sigma], dtype=dtype).to(magicClothingModel.load_device)
        timestep = real_sigma * 0
        latent_image=latent_image.to(magicClothingModel.load_device).to(dtype)
        # xc = magicClothingModel.model.model_sampling.calculate_input(real_sigma, latent_image).to(dtype)
        model_management.load_model_gpu(magicClothingModel)                      
        magicClothingModel.model.diffusion_model(latent_image, timestep, context=context, control=None, transformer_options=magicClothingModel.model_options["transformer_options"])
        comfy.sampler_helpers.cleanup_models({}, [magicClothingModel])
        return attn_stored

NODE_CLASS_MAPPINGS = {
    "Load Magic Clothing Model": LoadMagicClothingModel,
    "Add Magic Clothing Attention": AddMagicClothingAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Magic Clothing Model": "Load Magic Clothing Model",
    "Add Magic Clothing Attention": "Add Magic Clothing Attention",
}
