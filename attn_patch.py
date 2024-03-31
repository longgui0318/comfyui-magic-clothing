import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention

class InputPatch:
 
    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q,k,v)
        if "block" not in attn_stored:
            attn_stored["block"] = {}
        do_classifier_free_guidance = extra_options["do_classifier_free_guidance"]
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
            q = torch.cat([q, ref_q], dim=1)
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
        q, _ = torch.chunk(q, 2, dim=1)
        return q
    
    
    