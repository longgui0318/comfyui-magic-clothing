import copy
import torch
from safetensors import safe_open
from .process import load_seg_model, generate_mask
from .utils import is_torch2_available, prepare_image, prepare_mask
from diffusers import UNet2DConditionModel

if is_torch2_available():
    from .attention_processor import REFAttnProcessor2_0 as REFAttnProcessor
    from .attention_processor import AttnProcessor2_0 as AttnProcessor
    from .attention_processor import REFAnimateDiffAttnProcessor2_0 as REFAnimateDiffAttnProcessor
else:
    from .attention_processor import REFAttnProcessor, AttnProcessor


class ClothAdapter:
    def __init__(self, sd_pipe, ref_path, device, enable_cloth_guidance):
        self.enable_cloth_guidance = enable_cloth_guidance
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.set_adapter(self.pipe.unet, "write")

        ref_unet = copy.deepcopy(sd_pipe.unet)
        if ref_unet.config.in_channels == 9:
            ref_unet.conv_in = torch.nn.Conv2d(4, 320, ref_unet.conv_in.kernel_size, ref_unet.conv_in.stride, ref_unet.conv_in.padding)
            ref_unet.register_to_config(in_channels=4)
        state_dict = {}
        hash_v1 = {}
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        ref_unet.load_state_dict(state_dict, strict=False)
        self.ref_unet = ref_unet.to(self.device, dtype=self.pipe.dtype)
        self.set_adapter(self.ref_unet, "read")
        self.attn_store = {}

    def set_adapter(self, unet, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
            self,
            cloth_latent,
            gen_latents,
            prompt_embeds_null,
            positive=None,
            negative=None,
            num_images_per_prompt=4,
            seed=-1,
            guidance_scale=7.5,
            cloth_guidance_scale=2.5,
            num_inference_steps=20,
            height=512,
            width=384,
            **kwargs,
    ):
        if gen_latents is not None:
            gen_latents = 0.18215 * gen_latents
            gen_latents=gen_latents.to(self.device).to(dtype=self.pipe.dtype)
        cloth_latent=cloth_latent.to(self.device).to(dtype=self.pipe.dtype)
        prompt_embeds_null = prompt_embeds_null.to(self.device).to(dtype=self.pipe.dtype)
        positive = positive.to(self.device).to(dtype=self.pipe.dtype)
        negative = negative.to(self.device).to(dtype=self.pipe.dtype)
        with torch.inference_mode():
            cloth_latent.__hash_log__("latent_image")
            prompt_embeds_null.__hash_log__("context")
            cloth_latent = 0.18215 * cloth_latent
            self.ref_unet(torch.cat([cloth_latent] * num_images_per_prompt), 0, torch.cat([prompt_embeds_null] * num_images_per_prompt), cross_attention_kwargs={"attn_store": self.attn_store})

        self.generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        if self.enable_cloth_guidance:
            images = self.pipe(
                prompt_embeds=positive,
                negative_prompt_embeds=negative,
                guidance_scale=guidance_scale,
                cloth_guidance_scale=cloth_guidance_scale,
                num_inference_steps=num_inference_steps,
                latents = gen_latents,
                generator=self.generator,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0, "enable_cloth_guidance": self.enable_cloth_guidance},
                **kwargs,
            ).images
        else:
            images = self.pipe(
                prompt_embeds=positive,
                negative_prompt_embeds=negative,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                latents = gen_latents,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0, "enable_cloth_guidance": self.enable_cloth_guidance},
                **kwargs,
            ).images

        return images