import torch
import matplotlib.pyplot as plt
# 定义输入范围
x = torch.linspace(-6, 6, 500)
# 标准 Sigmoid
y_sigmoid = torch.sigmoid(x)
# Hard Sigmoid 自定义实现
def hard_sigmoid(x):
   return torch.clamp(x / 6 + 0.5, min=0., max=1.)
y_hard_sigmoid = hard_sigmoid(x)
# 绘制对比图
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_sigmoid.numpy(), label='Sigmoid', linewidth=2)
plt.plot(x.numpy(), y_hard_sigmoid.numpy(), label='Hard Sigmoid', linestyle='--', linewidth=2)
plt.grid()
plt.title('Sigmoid vs Hard Sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()