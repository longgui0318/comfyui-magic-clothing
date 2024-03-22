import torch
from torch import nn
from enum import Enum
import logging

import yaml

import comfy.utils
import comfy.model_management
import comfy.clip_vision
import comfy.gligen
import comfy.diffusers_convert
import comfy.model_base
import comfy.model_detection
import comfy.sd1_clip
import comfy.sd2_clip
import comfy.sdxl_clip
import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.supported_models_base
import comfy.taesd.taesd
import comfy.sd

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, Timestep
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from diffusers.models.attention_processor import AttnProcessor


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    # 加载模型检查点文件
    sd = comfy.utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    # 计算模型参数数量
    parameters = comfy.utils.calculate_parameters(sd, "model.diffusion_model.")
    # 获取设备信息
    load_device = comfy.model_management.get_torch_device()

    # 从检查点文件中提取模型配置信息
    model_config = comfy.model_detection.model_config_from_unet(sd, "model.diffusion_model.")
    unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    # 如果未检测到模型配置信息，则引发运行时错误
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    # 如果模型配置包含 CLIP 视觉前缀，则加载 CLIP 视觉模型
    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = comfy.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    # 如果需要输出模型，则加载模型
    if output_model:
        inital_load_device = comfy.model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = comfy.model_management.unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    # 如果需要输出 VAE 模型，则加载 VAE 模型
    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = comfy.sd.VAE(sd=vae_sd)

    # 如果需要输出 CLIP 模型，则加载 CLIP 模型
    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = comfy.sd.CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    logging.warning("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    # 输出未使用的键
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    # 如果需要输出模型，则创建模型 patcher
    if output_model:
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device(), current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            comfy.model_management.load_model_gpu(model_patcher)

    model_patcher.set_model_unet_function_wrapper
    # 返回加载的模型 patcher、CLIP 模型、VAE 模型和 CLIP 视觉模型
    return (model_patcher, clip, vae, clipvision)



class ClothAttBlock(nn.Module):
    """处理衣服并使用该注意力层模块

    """
    def __init__(self, c, c_hidden):
        super().__init__()
        # depthwise/attention
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(c, c, kernel_size=3, groups=c)
        )

        # channelwise
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )

        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=True)

        # Init weights
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def _norm(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        mods = self.gammas

        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        try:
            x = x + self.depthwise(x_temp) * mods[2]
        except: #operation not implemented for bf16
            x_temp = self.depthwise[0](x_temp.float()).to(x.dtype)
            x = x + self.depthwise[1](x_temp) * mods[2]

        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]

        return x



class OMSUnet(UNetModel):
    def __init__(self, levels=2, bottleneck_blocks=12, c_hidden=384, c_latent=4, codebook_size=8192):
        super().__init__()
        
        self.c_latent = c_latent
        c_levels = [c_hidden // (2 ** i) for i in reversed(range(levels))]

        # Encoder blocks
        self.in_block = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(3 * 4, c_levels[0], kernel_size=1)
        )
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = ClothAttBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(nn.Sequential(
            nn.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        ))
        self.down_blocks = nn.Sequential(*down_blocks)
        self.down_blocks[0]

        self.codebook_size = codebook_size
        # self.vquantizer = VectorQuantize(c_latent, k=codebook_size)

        # Decoder blocks
        up_blocks = [nn.Sequential(
            nn.Conv2d(c_latent, c_levels[-1], kernel_size=1)
        )]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ClothAttBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    nn.ConvTranspose2d(c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2,
                                       padding=1))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(c_levels[0], 3 * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def encode(self, x, quantize=False):
        x = self.in_block(x)
        x = self.down_blocks(x)
        if quantize:
            qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
            return qe, x, indices, vq_loss + commit_loss * 0.25
        else:
            return x

    def decode(self, x):
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss


class OMSModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, device=device,unet_model=OMSUnet)
        self.diffusion_model.eval().requires_grad_(False)