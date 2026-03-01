import torch

act_scales = {}

# 模拟跑 5 条数据，每次 q_proj 通道0 的最大值不同
data = [5.0, 3.0, 72.0, 10.0, 8.0]

for i, val in enumerate(data):
    comming_max = torch.tensor([val])
    name = 'q_proj'
    
    if name in act_scales:
        act_scales[name] = torch.max(act_scales[name], comming_max)
    else:
        act_scales[name] = comming_max
    
    print(f'第{i+1}条: comming_max={val}  ->  act_scales={act_scales[name].item()}')

print(f'最终: {act_scales[name].item()}  ← 5条数据里最大的就是72.0')