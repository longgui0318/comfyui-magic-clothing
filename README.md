# comfyui-magic-clothing

## Installation

use `ComfyUI-Manager` or put this code into `custom_nodes`


## For samples, please refer to (here)[./example.json]


# Note

* Currently there are still problems with the low success rate of some of the adopters, which doesn't work well for dense patterns, (meanwhile the [sigma] parameter serves as a temporary solution to the input scaling of the clothing feature in comfyui)
* 当前实现抽卡概率还不够，主要是对于第一次unet采样时model.model_sampling.calculate_input 处理问题。还在研究中，先释放一个版本