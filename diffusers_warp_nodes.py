import torch
import os
import folder_paths
from pathlib import Path

from comfy import model_management

from .diffusers_magic_clothing.garment_diffusion import ClothAdapter
from .diffusers_magic_clothing.MagicClothingDiffusionPipeline import MagicClothingDiffusionPipeline


class ChangePixelValueNormalization:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"pixels": ("IMAGE", ),
                 "mode": (["[0,1]=>[-1,1]", "[-1,1]=>[0,1]"],),
                 }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalization"

    CATEGORY = "image"

    def normalization(self, pixels, mode):
        if mode == "[0,1]=>[-1,1]":
            pixels = (pixels * 255).round().clamp(min=0, max=255) / 127.5 - 1.0
        elif mode == "[-1,1]=>[0,1]":
            pixels = ((pixels+1) * 127.5).clamp(min=0, max=255) / 255.0
        else:
            pixels = pixels
        return (pixels,)


class ChangePipelineDtypeAndDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"pipeline": ("PIPELINE", ),
                 "dtype": (["default", "float32", "float16", "bfloat16"],),
                 "device": (["default", "cpu", "cuda", "cuda:0", "cuda:1"],),
                 }
                }
    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "change_dtype"

    CATEGORY = "pipeline"

    def change_dtype(self, pipeline, dtype="default", device="default"):
        if dtype == "float16":
            seleted_type = torch.float16
        elif dtype == "bfloat16":
            seleted_type = torch.bfloat16
        else:
            seleted_type = torch.float32
        if device == "default":
            seleted_device = model_management.get_torch_device()
        else:
            seleted_device = torch.device(device)
        pipeline = pipeline.to(seleted_device, dtype=seleted_type)
        pipeline.device = seleted_device
        pipeline.dtype = seleted_type
        return (pipeline,)


class RunMagicClothingDiffusersModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cloth_image": ("IMAGE",),
                             "magicClothingAdapter": ("MAGIC_CLOTHING_ADAPTER",),
                             "positive": ("STRING", {
                                 "dynamicPrompts": False,
                                 "multiline": True,
                                 "default": ""
                             }),
                             "negative": ("STRING", {
                                 "dynamicPrompts": False,
                                 "multiline": True,
                                 "default": ""
                             }),
                             "height": ("INT", {"default": 768, "min": 0, "max": 2048}),
                             "width": ("INT", {"default": 576, "min": 0, "max": 2048}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                             "steps": ("INT", {"default": 25, "min": 0, "max": 100}),
                             "cfg": ("FLOAT", {"default": 5, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "cloth_guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                             }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_model"

    CATEGORY = "loaders"

    def run_model(self, cloth_image, magicClothingAdapter, positive, negative, height, width, batch_size, steps, cfg, cloth_guidance_scale, seed,):
        cloth_image = (cloth_image * 255).round().clamp(min=0,
                                                        max=255).to(dtype=torch.float32) / 127.5 - 1.0
        cloth_image = cloth_image.permute(0, 3, 1, 2)
        if not isinstance(magicClothingAdapter, ClothAdapter):
            # 如果发现不是正确的模型，就返回原始图片，不进行处理
            gen_image = cloth_image.permute(0, 2, 3, 1)
            gen_image = ((gen_image+1) * 127.5).clamp(min=0,
                                                      max=255).to(dtype=torch.float32) / 255.0
            return (gen_image,)
        magicClothingAdapter.enable_cloth_guidance = True
        cloth_image = cloth_image.to(
            magicClothingAdapter.pipe.device, dtype=magicClothingAdapter.pipe.dtype)
        with torch.inference_mode():
            prompt_embeds_null = magicClothingAdapter.pipe.encode_prompt(
                [""], device=magicClothingAdapter.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
            prompt_embeds, negative_prompt_embeds = magicClothingAdapter.pipe.encode_prompt(
                "a photography of a model,best quality, high quality",
                magicClothingAdapter.pipe.device,
                batch_size,
                True,
                "bare, monochrome, lowres, bad anatomy, worst quality, low quality",
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            cloth_latent = magicClothingAdapter.pipe.vae.encode(
                cloth_image).latent_dist.mode()
            gen_image = magicClothingAdapter.generate(cloth_latent, None, prompt_embeds_null, prompt_embeds, negative_prompt_embeds, batch_size, seed, cfg, cloth_guidance_scale, steps, height, width)
            gen_image = magicClothingAdapter.pipe.vae.decode(
                gen_image, return_dict=False, generator=magicClothingAdapter.generator)[0]
        gen_image = gen_image.permute(0, 2, 3, 1)
        gen_image = ((gen_image+1) * 127.5).clamp(min=0,
                                                  max=255).to(dtype=torch.float32) / 255.0
        return (gen_image,)


class LoadMagicClothingPipeline:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "dtype": (["default", "float32", "float16", "bfloat16"],),
                             "device": (["default", "cpu", "cuda", "cuda:0", "cuda:1"],), }}

    RETURN_TYPES = ("PIPELINE", "AUTOENCODER", "SCHEDULER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name,dtype,device):
        if dtype == "float16":
            seleted_type = torch.float16
        elif dtype == "bfloat16":
            seleted_type = torch.bfloat16
        else:
            seleted_type = torch.float32
        if device == "default":
            seleted_device = model_management.get_torch_device()
        else:
            seleted_device = torch.device(device)
        my_path = os.path.dirname(__file__)
        my_pipeline_path = os.path.join(my_path, "conversion")
        if not os.path.exists(my_pipeline_path):
            os.makedirs(my_pipeline_path)
        real_ckpt_name = Path(ckpt_name).stem
        real_ckpt_name = real_ckpt_name +"_"+str(seleted_type)
        real_ckpt_name = real_ckpt_name.replace(" ", "_").replace(".", "_").replace("/", "_")
        ckpt_conversion_path = os.path.join(my_pipeline_path, real_ckpt_name)
        if not os.path.exists(ckpt_conversion_path):
            # 不存在，则进行转换
            MagicClothingDiffusionPipeline.from_single_file(
                pretrained_model_link_or_path=folder_paths.get_full_path("checkpoints", ckpt_name),
                torch_dtype=seleted_type,
                cache_dir=my_pipeline_path,
            ).save_pretrained(ckpt_conversion_path, safe_serialization=True)

        pipe = MagicClothingDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt_conversion_path,
            torch_dtype=seleted_type,
            cache_dir=my_pipeline_path,
        )
        pipe.to(seleted_device, dtype=seleted_type)
        return ((pipe, ckpt_conversion_path), pipe.vae, pipe.scheduler)


class LoadMagicClothingAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"magicClothingUnet": (folder_paths.get_filename_list("unet"), ),
                 "pipeline": ("MAKED_PIPELINE", ),
                 },
                }

    RETURN_TYPES = ("MAGIC_CLOTHING_ADAPTER",)
    RETURN_NAMES = ("MagicClothingAdapter",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, magicClothingUnet, pipeline):
        unet_path = folder_paths.get_full_path("unet", magicClothingUnet)
        full_model = ClothAdapter(pipeline, unet_path)
        return (full_model,)


NODE_CLASS_MAPPINGS = {
    "Change Pixel Value Normalization": ChangePixelValueNormalization,
    "Change Pipeline Dtype And Device": ChangePipelineDtypeAndDevice,
    "Load Magic Clothing Pipeline": LoadMagicClothingPipeline,
    "Load Magic Clothing Adapter": LoadMagicClothingAdapter,
    "RUN Magic Clothing Diffusers Model": RunMagicClothingDiffusersModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Change Pipeline Dtype And Device": "Change Pipeline Dtype And Device",
    "Change Pixel Value Normalization": "Change Pixel Value Normalization",
    "Load Magic Clothing Pipeline":"Load Magic Clothing Pipeline&Diffusers",
    "Load Magic Clothing Adapter": "Load Magic Clothing Adapter &Diffusers",
    "RUN Magic Clothing Adapter": "RUN Magic Clothing Adapter &Diffusers",
}
