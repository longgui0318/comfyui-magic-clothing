import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention

class InputPatch:
    # forward for patching
    def __init__(self, type):
        self.type = type
 
    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q,k,v)
        if self.type == "save":
            if "block" not in attn_stored:
                attn_stored["block"] = {}
            block_name = extra_options["block"][0]
            block_id = extra_options["block"][1]
            if block_name not in attn_stored["block"]:
                attn_stored["block"][block_name] = {}
            attn_stored["block"][block_name][block_id] = q
        elif self.type == "restore":
            if "block" not in attn_stored:
                attn_stored["block"] = {}
            do_classifier_free_guidance = extra_options["do_classifier_free_guidance"]
            enable_cloth_guidance = extra_options["enable_cloth_guidance"]
            block_name = extra_options["block"][0]
            block_id = extra_options["block"][1]
            if block_name in attn_stored["block"] and block_id in attn_stored["block"][block_name]:
                ref_q = attn_stored["block"][block_name][block_id]
            if do_classifier_free_guidance:
                empty_copy = torch.zeros_like(ref_q)
                if enable_cloth_guidance:
                    ref_q = torch.cat([empty_copy, ref_q, ref_q])
                else:
                    ref_q = torch.cat([empty_copy, ref_q])
            q = torch.cat([q, ref_q], dim=1)
            return (q,q,q)
        return (q,k,v)

class OutputPatch:
    # forward for patching
    def __init__(self, type):
        self.type = type
 
    def __call__(self, q, k, v, extra_options):
        if "attn_stored" in extra_options:
            attn_stored = extra_options["attn_stored"]
        if attn_stored is None:
            return (q,k,v)
        if self.type == "restore":
             q, _ = torch.chunk(q, 2, dim=1)
             return (q,q,q)
        return (q,k,v)