import torch
import math
from typing import Any, Optional
import torch.nn.functional as F
from comfy import model_management
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0
from comfy.ldm.modules.attention import optimized_attention
from .utils import save_attn,clean_attn_stored_memory

class REFAttnProcessor(AttnProcessor):
    def __init__(self,need_save=True,block_name=None,block_number=None,attention_index=None):
        super().__init__()
        self.block_name = block_name
        self.block_number = block_number
        self.attention_index = attention_index
        self.need_save = need_save

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        attn_store=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.need_save :
            save_attn(hidden_states,attn_store,self.block_name,self.block_number,self.attention_index)
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )


class REFAttnProcessor2_0(AttnProcessor2_0):
    def __init__(self,need_save=True,block_name=None,block_number=None,attention_index=None):
        super().__init__()
        self.block_name = block_name
        self.block_number = block_number
        self.attention_index = attention_index
        self.need_save = need_save

    def __call__(
         self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        attn_store=None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.need_save :
            save_attn(hidden_states,attn_store,self.block_name,self.block_number,self.attention_index)
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )
        
class SamplerCfgFunctionWrapper:
        
    def _rescale_noise_cfg_(self,noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        *** copy from diffusers pipeline_stable_diffusion.py -> rescale_noise_cfg ***
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg
        
    def __call__(self, parameters) -> Any:
        cond = parameters["cond"]
        uncond = parameters["uncond"]
        input_x = parameters["input"]
        cond_scale = parameters["cond_scale"]
        model_options = parameters["model_options"]
        transformer_options = model_options["transformer_options"]
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            feature_guidance_scale = attn_stored["feature_guidance_scale"]
            cond_or_uncond_out_cond = attn_stored["cond_or_uncond_out_cond"]
            cond_or_uncond_out_count = attn_stored["cond_or_uncond_out_count"]
            #clear memory
            clean_attn_stored_memory(attn_stored)
            if  cond_or_uncond_out_cond is None:
                return uncond + (cond - uncond) * cond_scale
            else:
                cond = input_x - cond
                uncond = input_x - uncond
                cond_or_uncond_out_cond /= cond_or_uncond_out_count
                noise_pred = input_x - (
                    uncond
                    + cond_scale * (cond - cond_or_uncond_out_cond)
                    + feature_guidance_scale * (cond_or_uncond_out_cond - uncond))
                return self._rescale_noise_cfg_(noise_pred,cond,cond_scale)
        else:
            return uncond + (cond - uncond) * cond_scale
        
class UnetFunctionWrapper:
    
    def _is_inject_batch_(self,model,input,inject_batch_count):
        free_memory = model_management.get_free_memory(input.device)
        input_shape = [input[0] + inject_batch_count] + list(input)[1:]
        return model.memory_required(input_shape) < free_memory
    
    def __call__(self,apply_model, parameters):
        input = parameters["input"]
        timestep = parameters["timestep"]
        c = parameters["c"]                                     
        transformer_options = c["transformer_options"]
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            enable_feature_guidance = attn_stored["enable_feature_guidance"]
            input_x_extra_options = attn_stored["input_x_extra_options"]
            fined_nput_x_extra_option_indexs =  []
            cond_or_uncond = parameters["cond_or_uncond"]
            cond_or_uncond_replenishment = []
            #对传入参数进行调整，调整方式如下
            # A 对负向提示词，复制一份，这是为了计算出空数据的情况，插入的方式在前面
            # B 对正向忽略
            input_array =torch.chunk(input,input.shape[0])
            timestep_array = torch.chunk(timestep,timestep.shape[0])
            new_input_array = []
            new_timestep = []
            new_c_concat = None
            new_c_crossattn = None
            c_concat_array = None
            c_crossattn_array = None
            cond_or_uncond_extra_options = {}
            if "c_concat" in c:
                c_concat = c["c_concat"]
                c_concat_array = torch.chunk(c_concat,c_concat.shape[0])
                new_c_concat = []
            if "c_crossattn" in c:
                c_crossattn = c["c_crossattn"]
                c_crossattn_array = torch.chunk(c_crossattn,c_crossattn.shape[0])
                new_c_crossattn = []
            for i in range(len(input_array)):
                cond_flag = cond_or_uncond[i] # 需注意，3月底comfyui更新，为了支持多conds实现，移除了cond本身的判定，这个值存的是index
                fined_nput_x_extra_option = None
                for input_x_extra_i in range(len(input_x_extra_options)):
                    if input_x_extra_i in fined_nput_x_extra_option_indexs:
                        continue
                    fined_nput_x_extra_option_indexs.append(input_x_extra_i)
                    if torch.eq(input_array[i],input_x_extra_options[input_x_extra_i]["input_x"]).all():
                        fined_nput_x_extra_option = input_x_extra_options[input_x_extra_i]
                        break
                new_input_array.append(input_array[i])
                new_timestep.append(timestep_array[i])
                if c_concat_array is not None:
                    new_c_concat.append(c_concat_array[i])
                if c_crossattn_array is not None:
                    new_c_crossattn.append(c_crossattn_array[i])
                cond_or_uncond_replenishment.append(1 if cond_flag == 1 else 0)
                if enable_feature_guidance and cond_flag == 1:
                    cond_or_uncond_extra_options[i+1]={
                        "mult":fined_nput_x_extra_option["mult"] if fined_nput_x_extra_option is not None else None,
                        "area":fined_nput_x_extra_option["area"] if fined_nput_x_extra_option is not None else None
                    }
                    cond_or_uncond_replenishment.append(2)# 注意，在启用特征引导的时候，需要增加一个负向空特征来处理，这个复制的负向特征是给后面计算空特征用的
                    new_input_array.append(input_array[i])
                    new_timestep.append(timestep_array[i])
                    if c_concat_array is not None:
                        new_c_concat.append(c_concat_array[i])
                    if c_crossattn_array is not None:
                        new_c_crossattn.append(c_crossattn_array[i])
            input = torch.cat(new_input_array,)
            timestep = torch.cat(new_timestep,)
            if new_c_concat is not None:
                c["c_concat"] = torch.cat(new_c_concat,)
            if new_c_crossattn is not None:
                c["c_crossattn"] = torch.cat(new_c_crossattn,)
            if "out_cond_init" not in attn_stored:
                attn_stored["out_cond_init"] = torch.zeros_like(input_array[0])
            if "out_count_init" not in attn_stored:
                attn_stored["out_count_init"] = torch.zeros_like(input_array[0] * 1e-37)
            attn_stored["cond_or_uncond_replenishment"] = cond_or_uncond_replenishment
            attn_stored["cond_or_uncond_extra_options"] = cond_or_uncond_extra_options
            
            #直接清理，节省内存
            del input_array
            del timestep_array
            del new_input_array
            del new_timestep
            del new_c_concat
            del new_c_crossattn
            del c_concat_array
            del c_crossattn_array 
            del cond_or_uncond_extra_options
            for i in range(len(fined_nput_x_extra_option_indexs)- 1, -1, -1):
                del input_x_extra_options[fined_nput_x_extra_option_indexs[i]]
        
        output = apply_model(input,timestep,**c)
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            enable_feature_guidance = attn_stored["enable_feature_guidance"]
         
            cond_or_uncond_replenishment = attn_stored["cond_or_uncond_replenishment"]
            cond_or_uncond_extra_options = attn_stored["cond_or_uncond_extra_options"]
            pred_result = torch.chunk(output,len(cond_or_uncond_replenishment))
            new_output = []
            for i in range(len(cond_or_uncond_replenishment)):
                cond_flag = cond_or_uncond_replenishment[i]
                if cond_flag == 2:
                    cond_or_uncond_extra_option = cond_or_uncond_extra_options[i]
                    if "cond_or_uncond_out_cond" not in attn_stored:
                        attn_stored["cond_or_uncond_out_cond"] = attn_stored["out_cond_init"]
                    if "cond_or_uncond_out_count" not in attn_stored:
                        attn_stored["cond_or_uncond_out_count"] = attn_stored["out_count_init"]
                    mult = cond_or_uncond_extra_option["mult"]
                    area = cond_or_uncond_extra_option["area"]
                    
                    attn_stored["cond_or_uncond_out_cond"][:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] += pred_result[i] * mult
                    attn_stored["cond_or_uncond_out_count"][:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] += mult
                else:
                    new_output.append(pred_result[i])
            output = torch.cat(new_output)
            del new_output
            del pred_result
        return output


class SaveAttnInputPatch:
 
    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q,k,v)
        attn_stored_data = attn_stored["data"]
        block_name = extra_options["block"][0]
        block_id = extra_options["block"][1]
        block_index = extra_options["block_index"]
        if block_name not in attn_stored_data:
            attn_stored_data[block_name] = {}
        if block_id not in attn_stored_data[block_name]:
            attn_stored_data[block_name][block_id] = {}
        attn_stored_data[block_name][block_id][block_index] = q
        return (q,k,v)

class InputPatch:
 
    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q,k,v)
        attn_stored_data = attn_stored["data"]
        cond_or_uncond_replenishment = attn_stored["cond_or_uncond_replenishment"]
        block_name = extra_options["block"][0]
        block_id = extra_options["block"][1]
        block_index = extra_options["block_index"]
        if block_name in attn_stored_data and block_id in attn_stored_data[block_name] and block_index in attn_stored_data[block_name][block_id]:
            FLAG_OUT_CHANNEL = 2
            qEQk = q.shape[FLAG_OUT_CHANNEL] == k.shape[FLAG_OUT_CHANNEL]
            qEQv = q.shape[FLAG_OUT_CHANNEL] == v.shape[FLAG_OUT_CHANNEL]
            feature_hidden_states = attn_stored_data[block_name][block_id][block_index]
            if q.shape[1] != feature_hidden_states.shape[1]:
                clean_attn_stored_memory(attn_stored)
                raise ValueError("Your featured image must be the same width and height as the image you want to generate!")
            feature_hidden_states = feature_hidden_states.to(q.dtype)
            combo_feature_hidden_states = []
            for i in range(len(cond_or_uncond_replenishment)):
                cond_flag = cond_or_uncond_replenishment[i]
                if cond_flag == 0 or cond_flag == 2:
                    combo_feature_hidden_states.append(feature_hidden_states)
                else :
                    empty_feature = torch.zeros_like(feature_hidden_states)
                    combo_feature_hidden_states.append(empty_feature)
            feature_hidden_states = torch.cat(combo_feature_hidden_states)
            q = torch.cat([q, feature_hidden_states], dim=1)
            return (q,q if qEQk else k,q if qEQv else v)
        return (q,k,v)

class ReplacePatch:
 
    def __call__(self, q, k, v, extra_options):
        if extra_options is None:
            extra_options = {}
        n_heads = extra_options["n_heads"]
        q = optimized_attention(q, k, v, n_heads if n_heads is not None else 8)
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return q
        q, _ = torch.chunk(q, 2, dim=1)#抹除额外内容
        #对于整体的如何呢
        return q