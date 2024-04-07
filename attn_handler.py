import torch
import math
from typing import Any, Optional
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0
from comfy.ldm.modules.attention import optimized_attention
from .utils import save_attn

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
    
    def is_oms_mode(self,model_options):
        transformer_options = model_options["transformer_options"]
        return "attn_stored" in transformer_options
    
    def __call__(self, parameters) -> Any:
        cond = parameters["cond"]
        uncond = parameters["uncond"]
        cond_scale = parameters["cond_scale"]
        if self.is_oms_mode(parameters["model_options"]):
            return cond - uncond
        else:
            return uncond + (cond - uncond) * cond_scale
        
class UnetFunctionWrapper:
    def __call__(self,apply_model, parameters):
        input = parameters["input"]
        timestep = parameters["timestep"]
        c = parameters["c"]                                     
        transformer_options = c["transformer_options"]
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            enable_feature_guidance = attn_stored["enable_feature_guidance"]
            input_x_extra_options = attn_stored["input_x_extra_options"]
            un_handle_input_x_index =  list(range(len(input_x_extra_options)))
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
                find_input_x_index = None
                for un_handle_input_x_i in un_handle_input_x_index:
                    un_handle_input_x_item = un_handle_input_x_index[un_handle_input_x_i]
                    un_handle_input_x_item_input_x = un_handle_input_x_item["input_x"]
                    if torch.eq(input_array[i],un_handle_input_x_item_input_x).all():
                        find_input_x_index = un_handle_input_x_i
                        break
                find_input_x = None
                if find_input_x_index is not None:
                    find_input_x = un_handle_input_x_index[find_input_x_index]["input_x"]
                new_input_array.append(input_array[i])
                new_timestep.append(timestep_array[i])
                if c_concat_array is not None:
                    new_c_concat.append(c_concat_array[i])
                if c_crossattn_array is not None:
                    new_c_crossattn.append(c_crossattn_array[i])
                cond_or_uncond_replenishment.append(1 if cond_flag == 1 else 0)
                if enable_feature_guidance and cond_flag == 1:
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
            attn_stored["cond_or_uncond_replenishment"] = cond_or_uncond_replenishment
            
            #直接清理，节省内存
            del input_array
            del timestep_array
            del new_input_array
            del new_timestep
            del new_c_concat
            del new_c_crossattn
            del c_concat_array
            del c_crossattn_array 
        
        output = apply_model(input,timestep,**c)
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            enable_feature_guidance = attn_stored["enable_feature_guidance"]
            feature_guidance_scale = attn_stored["feature_guidance_scale"]
            cond_scale = attn_stored["cond_scale"]
            disable_cfg1_optimization = attn_stored["disable_cfg1_optimization"]            
            cond_or_uncond_replenishment = attn_stored["cond_or_uncond_replenishment"]
            # 根据oms的特征处理公式，简化成a-b的模式 同上面SamplerCfgFunctionWrapper的包装处理
            # 令空特征 结果为 nEqp 负向特征 结果为 nRqp 正向特征 结果为 pRqp 
            # 最终结果为 ( nEqp+cond_scale*pRqp + feature_guidance_scale*nRqp ) - ( cond_scale*nRqp + feature_guidance_scale * nEqp)
            pred_result = torch.chunk(output,len(cond_or_uncond_replenishment))
            nEqp = None
            nRqp = None
            pRqp = None
            for i in range(len(cond_or_uncond_replenishment)):
                cond_flag = cond_or_uncond_replenishment[i]
                
        # 输出数据，首选，根据是否为负向提示词，是，提取出前面的计算量，并缓存保存
        return output

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
            feature_hidden_states = attn_stored_data[block_name][block_id][block_index]
            feature_hidden_states = feature_hidden_states.to(q.dtype)
            combo_feature_hidden_states = []
            has_feature_guidance = 2 in cond_or_uncond_replenishment
            for i in range(len(cond_or_uncond_replenishment)):
                cond_flag = cond_or_uncond_replenishment[i]
                if cond_flag == 0 or (cond_flag == 1 and has_feature_guidance):
                    combo_feature_hidden_states.append(feature_hidden_states)
                else :
                    empty_feature = torch.zeros_like(feature_hidden_states)
                    combo_feature_hidden_states.append(empty_feature)
            feature_hidden_states = torch.cat(combo_feature_hidden_states)
            q = torch.cat([q, feature_hidden_states], dim=1)
            return (q,q,q)
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