from .oms_diffusion_nodes import NODE_CLASS_MAPPINGS as CM_O, NODE_DISPLAY_NAME_MAPPINGS as NM_O
from .diffusers_nodes import NODE_CLASS_MAPPINGS as CM_D, NODE_DISPLAY_NAME_MAPPINGS as NM_D

NODE_CLASS_MAPPINGS = {
    **CM_O, 
    # **CM_D
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    **NM_O,
    # **NM_D
    }
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']