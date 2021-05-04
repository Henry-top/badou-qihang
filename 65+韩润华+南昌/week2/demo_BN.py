import numpy as np
import torch


"""
实现batchnormalization层前向计算
"""
"""
1. 随机输入
2. 定义bn层
3. 生成新的权重(随机值)
4. 使用bn层进行前向计算（bn层就是一个普通的）
"""
# 1.准备一个随机输入
x = torch.randn(4, 8)   # 4个8维向量作为一个batch输入
# 2.定义bn层，参数要与输入的维度一致
bn = torch.nn.BatchNorm1d(8)
# 3.随机生成一个新的权重代替初始权重weight
weight = torch.randn(bn.state_dict()["weight"].shape)
bn.weight = torch.nn.Parameter(weight)
# 看看改变后的权重输出。
# 如果是原始的初始化权重，应该是[1,1,1,...]
print(bn.state_dict()["weight"], "bn层权重")
# 4.使用bn层进行前向计算
y = bn(x)
print(y, "torch bn输出")


# # 取出参数
# w = bn.state_dict()["weight"].numpy()
# b = bn.state_dict()["bias"].numpy()
#
# # 将输入转成numpy数组
# x = x.numpy()
# # 计算均值，注意:沿batch_size的维度(就是有几个输入向量)进行均值计算
# p = np.mean(x, axis=0)
# # print(p)
# # 按照公式计算 var
# v = np.mean(np.square(x - p), axis=0)       # square 计算各个元素的平方
# # 依然是按照公式计算，这里e=1e-5是为了防止分母为零
# x = (x - p) / np.sqrt(v + 1e-5)     # sqrt() 计算各个元素的平方根
# # 最后的scale线性运算
# y = w * x + b
# print(y, "自定义bn输出")
