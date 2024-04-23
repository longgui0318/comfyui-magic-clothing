import torch
import folder_paths

from .utils import pt_hash

from diffusers import UniPCMultistepScheduler,AutoencoderKL

from .oms.garment_diffusion import ClothAdapter
from .oms.OmsDiffusionPipeline import OmsDiffusionPipeline
    
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


class RunOmsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cloth_image": ("IMAGE",),
                             "model": ("MODEL",),
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
        cloth_image =  (cloth_image * 255).round().clamp(min=0,max=255).to(dtype=torch.float32)  / 127.5 - 1.0
        cloth_image =  cloth_image.permute(0,3,1,2)
        if not isinstance(model,ClothAdapter):
            gen_image = cloth_image.permute(0, 2, 3, 1)
            gen_image = ((gen_image+1) * 127.5).clamp(min=0,max=255).to(dtype=torch.float32) / 255.0
            return (gen_image,)
        cloth_image =  cloth_image.to(model.device, dtype=model.pipe.dtype)
        with torch.inference_mode():
            prompt_embeds_null = model.pipe.encode_prompt([""], device=model.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
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
        gen_image = gen_image.permute(0, 2, 3, 1)
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
        if "768" not in magicClothingUnet:
            return ({},)
        unet_path = folder_paths.get_full_path("unet", magicClothingUnet)
        pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float32)
        pipe = OmsDiffusionPipeline.from_pretrained(pipe_path,vae=vae,torch_dtype=torch.float32)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        full_net = ClothAdapter(pipe, unet_path, "cuda",True)
        return (full_net,)

NODE_CLASS_MAPPINGS = {
    "Change Pixel Value Normalization":ChangePixelValueNormalization,
    "RUN OMS": RunOmsNode,
    "LOAD OMS": LoadOmsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Change Pixel Value Normalization":"Change Pixel Value Normalization",
    "RUN OMS": "RUN OMS",
    "LOAD OMS": "LOAD OMS",
}
