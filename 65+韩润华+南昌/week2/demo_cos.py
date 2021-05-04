"""
把两个向量想象成空间的两条线段，都是从原点出发（[0, 0, ...]）,指向不同的方向。
两条线段之间形成一个夹角，如果夹角为0°，则方向相同、线段重合；
如果夹角为90°，则形成直角，方向完全不相似；
如果夹角为180°，则方向正好相反。
夹角越小，就代表越相似。
"""
import numpy as np


def cos_similar(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量a
    :param vector_b: 向量b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)  # * 表示数量积，dot表示矢量乘法

    # np.linalg.norm() 求范数，默认是二范数（ord = 2）
    denum = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    
    cos = num / denum
    sim = 0.5 + 0.5 * cos   # 归一化
    return sim
    # return cos


a = [1, 2, 3]
b = [1, 2, 4]

print(cos_similar(a, b))
