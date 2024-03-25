import torch
import copy
import inspect
import logging
import uuid
import folder_paths

import comfy.ops
import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.supported_models_base
import comfy.taesd.taesd
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.utils
import comfy.sample
import comfy.samplers

# refer ComfyUI_IPAdapter_plus.IPAdapterPlus.set_model_patch_replace
def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    # if "patches_replace" not in to:
    #     to["patches_replace"] = {}
    # if "attn2" not in to["patches_replace"]:
    #     to["patches_replace"]["attn2"] = {}
    # if key not in to["patches_replace"]["attn2"]:
    #     to["patches_replace"]["attn2"][key] = CrossAttentionPatch(**patch_kwargs)
    # else:
    #     to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def text_encoder(clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]
class ExtractFeaturesWithAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "clip": ("CLIP", ),
                 "feature_image": ("LATENT", ),
                 }
                }
    RETURN_TYPES = ("MODEL", "LATENT",)
    FUNCTION = "unet_extract_features"

    CATEGORY = "loaders"
    

    def unet_extract_features(self, model,clip, feature_image):
        
        empty_text = text_encoder(clip, "")
        print(id(model))
        once_model_patcher = model.clone()
        print(id(once_model_patcher))
        if once_model_patcher.model.diffusion_model.in_channels ==9:
            # if in_channels == 9, then we need to change the model to a new model
            print(id(once_model_patcher.model))
            once_model_patcher.model =copy.copy(once_model_patcher.model)
            print(id(once_model_patcher.model))
            # just support UnetModel
            if not isinstance(once_model_patcher.model.diffusion_model, comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel):
                raise Exception("Only support UnetModel")
            once_model_patcher.model.diffusion_model = comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel()
            if not once_model_patcher.model.model_config.get("disable_unet_model_creation", False):
                if once_model_patcher.model.model_config.manual_cast_dtype is not None:
                    operations = comfy.ops.manual_cast
                else:
                    operations = comfy.ops.disable_weight_init
            unet_config = copy.deepcopy(once_model_patcher.model.unet_config)
            unet_config.in_channels = 4
            unet_config.out_channels = 320
            once_model_patcher.model.diffusion_model = comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel(**unet_config, device=once_model_patcher.model.diffusion_model.device, operations=operations)
            once_model_patcher.model.diffusion_model.load_state_dict(model.model.diffusion_model.state_dict())
        
         
        latent_image = feature_image["samples"]
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

        noise_mask = None
        if "noise_mask" in feature_image:
            noise_mask = feature_image["noise_mask"]

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(once_model_patcher, noise, 1, 1,
                                      comfy.samplers.KSampler.SAMPLERS[0],
                                      comfy.samplers.KSampler.SCHEDULERS[0],
                                      empty_text, empty_text, latent_image,
                                      denoise=1, disable_noise=True, start_step=None, last_step=None,
                                      force_full_denoise=False,
                                      noise_mask=noise_mask, callback=None,
                                      disable_pbar=disable_pbar, seed=-1)

        # 设置 模型执行的监听器
        # 通过监听回调更改数据
        # 执行unet模型保存注意力特征
        # 通过附加器处理原始的model，然后返回原来的model

        
        return model, feature_image



NODE_CLASS_MAPPINGS = {
    "Extract Features With Attention": ExtractFeaturesWithAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract Features With Attention": "Extract Features With Attention",
}
