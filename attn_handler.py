import torch
from typing import Optional
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0
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
