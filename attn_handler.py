import torch
from typing import Any
from comfy import model_management
from comfy.ldm.modules.attention import optimized_attention
from .utils import clean_attn_stored_memory

class SamplerCfgFunctionWrapper:

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
            # clear memory
            clean_attn_stored_memory(attn_stored)
            if cond_or_uncond_out_cond is None:
                return uncond + (cond - uncond) * cond_scale
            else:
                cond = input_x - cond
                uncond = input_x - uncond
                cond_or_uncond_out_cond /= cond_or_uncond_out_count
                noise_pred = (
                    uncond
                    + cond_scale * (cond - cond_or_uncond_out_cond)
                    + feature_guidance_scale *
                    (cond_or_uncond_out_cond - uncond)
                )
                return input_x - noise_pred
        else:
            return uncond + (cond - uncond) * cond_scale


class UnetFunctionWrapper:

    def _is_inject_batch_(self, model, input, inject_batch_count):
        free_memory = model_management.get_free_memory(input.device)
        input_shape = [input[0] + inject_batch_count] + list(input)[1:]
        return model.memory_required(input_shape) < free_memory

    def _reorganization_c_data_(self,c,key):
        if key in c:
            return self._chunk_data_(c[key]) 
        return None,None
    
    def _chunk_data_(self,data):
        if data is None:
            return None,None
        return torch.chunk(data,data.shape[0]),[]

    def __call__(self, apply_model, parameters):
        input = parameters["input"]
        timestep = parameters["timestep"]
        c = parameters["c"]
        transformer_options = c["transformer_options"]
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            enable_feature_guidance = attn_stored["enable_feature_guidance"]
            cond_or_uncond = parameters["cond_or_uncond"]
            cond_or_uncond_replenishment = []
            # 对传入参数进行调整，调整方式如下
            # A 对负向提示词，复制一份，这是为了计算出空数据的情况，插入的方式在前面
            # B 对正向忽略
            input_array = torch.chunk(input, input.shape[0])
            timestep_array = torch.chunk(timestep, timestep.shape[0])
            new_input_array = []
            new_timestep = []
            
            c_concat_data,c_concat_data_new = self._reorganization_c_data_(c,"c_concat")
            c_crossattn_data,c_crossattn_data_new = self._reorganization_c_data_(c,"c_crossattn")
            c_attn_stored_mult_data,_ = self._reorganization_c_data_(c,"c_attn_stored_mult")
            c_attn_stored_area_data = c["c_attn_stored_area"] if "c_attn_stored_area" in c else None
            c_attn_stored_control_data = c["c_attn_stored_control"] if "c_attn_stored_control" in c else None
            #移除因为注入增加的内容，后续已不再需要
            c["c_attn_stored_mult"] = None
            c["c_attn_stored_area"] = None
            c["c_attn_stored_control"] = None
  
            cond_or_uncond_extra_options = {}
            for i in range(len(input_array)):
                # 需注意，3月底comfyui更新，为了支持多conds实现，移除了cond本身的判定，这个值存的是index
                cond_flag = cond_or_uncond[i]
                new_input_array.append(input_array[i])
                new_timestep.append(timestep_array[i])
                if c_concat_data is not None:
                    c_concat_data_new.append(c_concat_data[i])
                if c_crossattn_data is not None:
                    c_crossattn_data_new.append(c_crossattn_data[i])
                
                cond_or_uncond_replenishment.append(1 if cond_flag == 1 else 0)
                if enable_feature_guidance and cond_flag == 1:
                    
                    if c_attn_stored_mult_data is not None and  c_attn_stored_area_data is not None:
                        mult = c_attn_stored_mult_data[i]
                        area = c_attn_stored_area_data[i]
                        cond_or_uncond_extra_options[i+1] = {
                            "mult": mult.squeeze(0),
                            "area": area
                        }
                    # 注意，在启用特征引导的时候，需要增加一个负向空特征来处理，这个复制的负向特征是给后面计算空特征用的
                    cond_or_uncond_replenishment.append(2)
                    new_input_array.append(input_array[i])
                    new_timestep.append(timestep_array[i])
                    if c_concat_data is not None:
                        c_concat_data_new.append(c_concat_data[i])
                    if c_crossattn_data is not None:
                        c_crossattn_data_new.append(c_crossattn_data[i])
            input = torch.cat(new_input_array,)
            timestep = torch.cat(new_timestep,)
            if c_concat_data_new is not None:
                c["c_concat"] = torch.cat(c_concat_data_new,)
            if c_crossattn_data_new is not None:
                c["c_crossattn"] = torch.cat(c_crossattn_data_new,)
            if "out_cond_init" not in attn_stored:
                attn_stored["out_cond_init"] = torch.zeros_like(input_array[0])
            if "out_count_init" not in attn_stored:
                attn_stored["out_count_init"] = torch.zeros_like(input_array[0] * 1e-37)
            if c_attn_stored_control_data is not None:
                c['control'] = c_attn_stored_control_data.get_control(input, timestep, c, len(cond_or_uncond_replenishment))
            attn_stored["cond_or_uncond_replenishment"] = cond_or_uncond_replenishment
            attn_stored["cond_or_uncond_extra_options"] = cond_or_uncond_extra_options

            # 直接清理，节省内存
            del input_array
            del timestep_array
            del new_input_array
            del new_timestep
            del c_concat_data
            del c_concat_data_new
            del c_crossattn_data
            del c_crossattn_data_new
            del c_attn_stored_mult_data
            del c_attn_stored_area_data
            del cond_or_uncond_extra_options

        output = apply_model(input, timestep, **c)
        if "attn_stored" in transformer_options:
            attn_stored = transformer_options["attn_stored"]
            enable_feature_guidance = attn_stored["enable_feature_guidance"]

            cond_or_uncond_replenishment = attn_stored["cond_or_uncond_replenishment"]
            cond_or_uncond_extra_options = attn_stored["cond_or_uncond_extra_options"]
            pred_result = torch.chunk(
                output, len(cond_or_uncond_replenishment))
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
                    if area is None:
                        attn_stored["cond_or_uncond_out_cond"] += pred_result[i] * mult
                        attn_stored["cond_or_uncond_out_count"] += mult
                    else:
                        out_c = attn_stored["cond_or_uncond_out_cond"]
                        out_cts = attn_stored["cond_or_uncond_out_count"]
                        dims = len(area) // 2
                        for i in range(dims):
                            out_c = out_c.narrow(i + 2, area[i + dims], area[i])
                            out_cts = out_cts.narrow(i + 2, area[i + dims], area[i])
                        out_c += pred_result[i] * mult
                        out_cts += mult
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
            return (q, k, v)
        attn_stored_data = attn_stored["data"]
        block_name = extra_options["block"][0]
        block_id = extra_options["block"][1]
        block_index = extra_options["block_index"]
        if block_name not in attn_stored_data:
            attn_stored_data[block_name] = {}
        if block_id not in attn_stored_data[block_name]:
            attn_stored_data[block_name][block_id] = {}
        attn_stored_data[block_name][block_id][block_index] = q
        return (q, k, v)


def _check_(calc_sigmas,sigma):
    if calc_sigmas is None:
        return True
    for i in range(len(calc_sigmas)):
        if abs(calc_sigmas[i] - sigma.item()) < 0.000001:
            return True
    return False

class InputPatch:
    
    def _calculate_input_(hideen_states, sigma):
        return hideen_states / (sigma ** 2 + 1) ** 0.5

    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q, k, v)
        attn_stored_data = attn_stored["data"]
        cond_or_uncond_replenishment = attn_stored["cond_or_uncond_replenishment"]
        block_name = extra_options["block"][0]
        block_id = extra_options["block"][1]
        block_index = extra_options["block_index"]
        sigma = extra_options["sigmas"]
        calc_sigmas = attn_stored.get("calc_sigmas",None)
        if _check_(calc_sigmas,sigma) and block_name in attn_stored_data and block_id in attn_stored_data[block_name] and block_index in attn_stored_data[block_name][block_id]:
            FLAG_OUT_CHANNEL = 2
            qEQk = q.shape[FLAG_OUT_CHANNEL] == k.shape[FLAG_OUT_CHANNEL]
            qEQv = q.shape[FLAG_OUT_CHANNEL] == v.shape[FLAG_OUT_CHANNEL]
            feature_hidden_states = attn_stored_data[block_name][block_id][block_index]
            # feature_hidden_states = self._calculate_input_(feature_hidden_states, sigma)
            if q.shape[1] != feature_hidden_states.shape[1]:
                clean_attn_stored_memory(attn_stored)
                raise ValueError(
                    "Your featured image must be the same width and height as the image you want to generate!")
            feature_hidden_states = feature_hidden_states.to(q.dtype)
            combo_feature_hidden_states = []
            for i in range(len(cond_or_uncond_replenishment)):
                cond_flag = cond_or_uncond_replenishment[i]
                if cond_flag == 0 or cond_flag == 2:
                    combo_feature_hidden_states.append(feature_hidden_states)
                else:
                    empty_feature = torch.zeros_like(feature_hidden_states)
                    combo_feature_hidden_states.append(empty_feature)
            feature_hidden_states = torch.cat(combo_feature_hidden_states)
            q = torch.cat([q, feature_hidden_states], dim=1)
            return (q, q if qEQk else k, q if qEQv else v)
        return (q, k, v)


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
        sigma = extra_options["sigmas"]
        calc_sigmas = attn_stored.get("calc_sigmas",None)
        if _check_(calc_sigmas,sigma):
            q, _ = torch.chunk(q, 2, dim=1)  # 抹除额外内容
        return q
