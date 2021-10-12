# Jittor-MLP
Unofficial Implementation of MLP-Mixer, gMLP, resMLP, Vision Permutator, S2MLPv2, ConvMLP, ConvMixer in Jittor.



* Jittor and Pytorch implementaion of [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).

![](imgs/mlpmixer.png)

* Jittor and Pytorch implementaion of [VISION PERMUTATOR: A PERMUTABLE MLP-LIKE ARCHITECTURE FOR VISUAL RECOGNITION](https://arxiv.org/pdf/2106.12368).

![](imgs/vip.png)



* Jittor and Pytorch implementaion of [gMLP](https://arxiv.org/abs/2105.08050)

![](imgs/gmlp.png)

* Jittor and Pytorch implementaion of [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404).

![](imgs/resmlp.png)

* Jittor and Pytorch implementaion of [S2-MLPv2: Improved Spatial-Shift MLP Architecture for Vision](https://arxiv.org/abs/2108.01072).

  ![](imgs/s2mlpv2.png)

* Jittor and Pytorch implementaion of [ConvMixer: Patches Are All You Need?](https://openreview.net/pdf?id=TVHS5Y4dNvM).

![](imgs/convmixer.png)

* Jittor and Pytorch implementaion of [ConvMLP: Hierarchical Convolutional MLPs for Vision](https://arxiv.org/abs/2109.04454).

![](imgs/convmlp.png)




## Usage

```python
import jittor as jt
from models_jittor import gMLPForImageClassification as gMLP_jt
from models_jittor import ResMLPForImageClassification as ResMLP_jt
from models_jittor import MLPMixerForImageClassification as MLPMixer_jt
from models_jittor import ViP as ViP_jt
from models_jittor import S2MLPv2 as S2MLPv2_jt
from models_jittor import ConvMixer as ConvMixer_jt
from models_jittor import convmlp_s as ConvMLP_s_jt 
from models_jittor import convmlp_l as ConvMLP_l_jt 
from models_jittor import convmlp_m as ConvMLP_m_jt 

model_jt = MLPMixer_jt(
    image_size=(224,112),
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    d_model=256,
    depth=12,
)

images = jt.randn(8, 3, 224, 224)
with jt.no_grad():
    output = model_jt(images)
print(output.shape) # （8， 1000）

############################################################################

import torch
from models_pytorch import gMLPForImageClassification as gMLP_pt
from models_pytorch import ResMLPForImageClassification as ResMLP_pt
from models_pytorch import MLPMixerForImageClassification as MLPMixer_pt
from models_pytorch import ViP as ViP_pt
from models_pytorch import S2MLPv2 as S2MLPv2_pt 
from models_pytorch import ConvMixer as ConvMixer_pt 
from models_pytorch import convmlp_s as ConvMLP_s_pt 
from models_pytorch import convmlp_l as ConvMLP_l_pt 
from models_pytorch import convmlp_m as ConvMLP_m_pt 

model_pt = ViP_pt(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    d_model=256,
    depth=30,
    segments = 16,
    weighted = True
)

images = torch.randn(8, 3, 224, 224)

with torch.no_grad():
    output = model_pt(images)
print(output.shape) # （8， 1000）


############################## Non-square images and patch sizes #########################

model_jt = ViP_jt(
    image_size=(224, 112),
    patch_size=(16, 8),
    in_channels=3,
    num_classes=1000,
    d_model=256,
    depth=30,
    segments = 16,
    weighted = True
)
images = jt.randn(8, 3, 224, 112)
with jt.no_grad():
    output = model_jt(images)
print(output.shape) # （8， 1000）

############################## 2 Stages S2MLPv2 #########################
model_pt = S2MLPv2_pt(
    in_channels = 3,
    image_size = (224,224),
    patch_size = [(7,7), (2,2)],
    d_model = [192, 384],
    depth = [4, 14],
    num_classes = 1000, 
    expansion_factor = [3, 3]
)
```



## Citations

```bibtex
@misc{tolstikhin2021mlpmixer,
    title   = {MLP-Mixer: An all-MLP Architecture for Vision},
    author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    year    = {2021},
    eprint  = {2105.01601},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{hou2021vision,
    title   = {Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition},
    author  = {Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng},
    year    = {2021},
    eprint  = {2106.12368},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{liu2021pay,
  title={Pay Attention to MLPs},
  author={Liu, Hanxiao and Dai, Zihang and So, David R and Le, Quoc V},
  journal={arXiv preprint arXiv:2105.08050},
  year={2021}
}
```

```bibtex
@article{touvron2021resmlp,
  title={Resmlp: Feedforward networks for image classification with data-efficient training},
  author={Touvron, Hugo and Bojanowski, Piotr and Caron, Mathilde and Cord, Matthieu and El-Nouby, Alaaeldin and Grave, Edouard and Joulin, Armand and Synnaeve, Gabriel and Verbeek, Jakob and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2105.03404},
  year={2021}
}
```

```bibtex
@article{yu2021s,
  title={S $\^{} 2$-MLPv2: Improved Spatial-Shift MLP Architecture for Vision},
  author={Yu, Tan and Li, Xu and Cai, Yunfeng and Sun, Mingming and Li, Ping},
  journal={arXiv preprint arXiv:2108.01072},
  year={2021}
}
```

```bibtex
@article{li2021convmlp,
  title={ConvMLP: Hierarchical Convolutional MLPs for Vision},
  author={Li, Jiachen and Hassani, Ali and Walton, Steven and Shi, Humphrey},
  journal={arXiv preprint arXiv:2109.04454},
  year={2021}
}
```

