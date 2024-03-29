import torch
from typing import Optional
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0


class REFAttnProcessor(AttnProcessor):
    def __init__(self, name, type="read"):
        super().__init__()
        self.name = name
        self.type = type

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
        if self.type == "read" and attn_store is not None:
            attn_store[self.name] = hidden_states
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
    def __init__(self, name, type="read"):
        super().__init__()
        self.name = name
        self.type = type

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
        if self.type == "read" and attn_store is not None:
            attn_store[self.name] = hidden_states
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )
