# comfyui-magic-clothing

The comfyui supported version of the [Magic Clothing](https://github.com/ShineChen1024/MagicClothing) project, not the diffusers version, allows direct integration with modules such as ipadapter

## Installation

* use `ComfyUI-Manager` or put this code into `custom_nodes`
* Go to [huggingface](https://huggingface.co/ShineChen1024/MagicClothing) to download the models and move them to the `comfyui/models/unet` folder

## For samples, please refer to [here](./example.json)
## For ipadapter samples, please refer to [here](./ipadapter.json)


# Note

* Currently there are still problems with the low success rate of some of the adopters, which doesn't work well for dense patterns, (meanwhile the [sigma] parameter serves as a temporary solution to the input scaling of the clothing feature in comfyui)
* 当前实现抽卡概率还不够，主要是对于第一次unet采样时model.model_sampling.calculate_input 处理问题。还在研究中，先释放一个版本
