import jittor as jt
from models_jittor import gMLPForImageClassification as gMLP_jt
from models_jittor import ResMLPForImageClassification as ResMLP_jt
from models_jittor import MLPMixerForImageClassification as MLPMixer_jt
from models_jittor import ViP as ViP_jt
from models_jittor import S2MLPv2 as S2MLPv2_jt
from models_jittor import ConvMixer as ConvMixer_jt

import torch
from models_pytorch import gMLPForImageClassification as gMLP_pt
from models_pytorch import ResMLPForImageClassification as ResMLP_pt
from models_pytorch import MLPMixerForImageClassification as MLPMixer_pt
from models_pytorch import ViP as ViP_pt
from models_pytorch import S2MLPv2 as S2MLPv2_pt 
from models_pytorch import ConvMixer as ConvMixer_pt 
import time


import numpy as np
jt.flags.use_cuda = 1

bs = 32
test_img = np.random.random((bs,3,224,224)).astype('float32')

# 定义 pytorch & jittor 输入矩阵
pytorch_test_img = torch.Tensor(test_img).cuda()
jittor_test_img = jt.array(test_img)

# 跑turns次前向求平均值
turns = 100

model_name = "ConvMixer"

# 定义 pytorch & jittor 的xxx模型，如vgg
if model_name == "MLPMixer":
    pytorch_model = MLPMixer_pt(
        image_size=(224,224),
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        depth=12,
    ).cuda()
    jittor_model = MLPMixer_jt(
        image_size=(224,224),
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        depth=12,
    )
elif model_name == "gMLP":
    pytorch_model = gMLP_pt(
        image_size=(224,224),
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        d_ffn=1536,
        depth=30
    ).cuda()
    jittor_model = gMLP_jt(
        image_size=(224,224),
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        d_ffn=1536,
        depth=30
    )
elif model_name == "ResMLP":
    pytorch_model = ResMLP_pt(
        in_channels = 3,
        image_size = (224,224),
        patch_size = 16,
        d_model = 384,
        depth = 12,
        num_classes = 1000, 
        expansion_factor = 4
    ).cuda()
    jittor_model = ResMLP_jt(
        in_channels = 3,
        image_size = (224,224),
        patch_size = 16,
        d_model = 384,
        depth = 12,
        num_classes = 1000, 
        expansion_factor = 4
    )
elif model_name == "ViP":
    pytorch_model = ViP_pt(
        image_size=(224, 224),
        patch_size=(16, 8),
        in_channels=3,
        num_classes=1000,
        d_model=256,
        depth=30,
        segments = 16,
        weighted = True
    ).cuda()
    jittor_model = ViP_jt(
        image_size=(224, 224),
        patch_size=(16, 8),
        in_channels=3,
        num_classes=1000,
        d_model=256,
        depth=30,
        segments = 16,
        weighted = True
    )
elif model_name == 'ConvMixer':
    pytorch_model = ConvMixer_pt(
        dim = 1568,
        depth = 20
    ).cuda()
    jittor_model = ConvMixer_jt(
        dim = 1568,
        depth = 20
    )
else:
    pytorch_model = S2MLPv2_pt(
        in_channels = 3,
        image_size = (224,224),
        patch_size = [(7,7), (2,2)],
        d_model = [192, 384],
        depth = [4, 14],
        num_classes = 1000, 
        expansion_factor = [3, 3]
    ).cuda()
    jittor_model = S2MLPv2_jt(
        in_channels = 3,
        image_size = (224,224),
        patch_size = [(7,7), (2,2)],
        d_model = [192, 384],
        depth = [4, 14],
        num_classes = 1000, 
        expansion_factor = [3, 3]
    )

# 把模型都设置为eval来防止dropout层对输出结果的随机影响
pytorch_model.eval()
jittor_model.eval()

# jittor加载pytorch的初始化参数来保证参数完全相同
jittor_model.load_parameters(pytorch_model.state_dict())


# 测试Pytorch一次前向传播的平均用时
for i in range(10):
    pytorch_result = pytorch_model(pytorch_test_img) # Pytorch热身
torch.cuda.synchronize()
sta = time.time()
for i in range(turns):
    pytorch_result = pytorch_model(pytorch_test_img)
torch.cuda.synchronize() # 只有运行了torch.cuda.synchronize()才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
end = time.time()
tc_time = round((end - sta) / turns, 5) # 执行turns次的平均时间，输出时保留5位小数
tc_fps = round(bs * turns / (end - sta),0) # 计算FPS
print(f"- Pytorch {model_name} forward average time cost: {tc_time}, Batch Size: {bs}, FPS: {tc_fps}")


# 测试Jittor一次前向传播的平均用时
for i in range(10):
    jittor_result = jittor_model(jittor_test_img) # Jittor热身
    jittor_result.sync()
jt.sync_all(True)
# sync_all(true)是把计算图发射到计算设备上，并且同步。只有运行了jt.sync_all(True)才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
sta = time.time()
for i in range(turns):
    jittor_result = jittor_model(jittor_test_img)
    jittor_result.sync() # sync是把计算图发送到计算设备上
jt.sync_all(True)
end = time.time()
jt_time = round((time.time() - sta) / turns, 5) # 执行turns次的平均时间，输出时保留5位小数
jt_fps = round(bs * turns / (end - sta),0) # 计算FPS
print(f"- Jittor {model_name} forward average time cost: {jt_time}, Batch Size: {bs}, FPS: {jt_fps}")


threshold = 1e-3
# 计算 pytorch & jittor 前向结果相对误差. 如果误差小于threshold，则测试通过.
x = pytorch_result.detach().cpu().numpy() + 1
y = jittor_result.data + 1
relative_error = abs(x - y) / abs(y)
diff = relative_error.mean()
assert diff < threshold, f"[*] {model_name} forward fails..., Relative Error: {diff}"
print(f"[*] {model_name} forword passes with Relative Error {diff}")