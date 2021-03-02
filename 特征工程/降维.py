# -*- encoding: utf-8 -*-
"""
@File    : 降维.py
@Time    : 2019-04-10 17:17
@Author  : final
@Email   : julyfinal@outlook.com
@Software: PyCharm
"""

from sklearn.datasets import load_iris
# 导入IRIS数据集
iris = load_iris()

from sklearn.decomposition import PCA,LatentDirichletAllocation
# 主成分分析法，返回降维后的数据
# 参数n_components为主成分数目
PCA(n_components=2).fit_transform(iris.data)

#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
LatentDirichletAllocation(n_components=2).fit_transform(iris.data, iris.target)