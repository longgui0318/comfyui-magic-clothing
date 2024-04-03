import torch
from typing import Optional
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

class InputPatch:
 
    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q,k,v)
        do_classifier_free_guidance = False
        enable_cloth_guidance = extra_options["enable_cloth_guidance"]
        block_name = extra_options["block"][0]
        block_id = extra_options["block"][1]
        block_index = extra_options["block_index"]
        if block_name in attn_stored and block_id in attn_stored[block_name] and block_index in attn_stored[block_name][block_id]:
            ref_q = attn_stored[block_name][block_id][block_index]
            if do_classifier_free_guidance:
                empty_copy = torch.zeros_like(ref_q)
                if enable_cloth_guidance:
                    ref_q = torch.cat([empty_copy, ref_q, ref_q])
                else:
                    ref_q = torch.cat([empty_copy, ref_q])
            q = torch.cat([q, ref_q], dim=1)#参与计算
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