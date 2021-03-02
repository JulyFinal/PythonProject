# -*- encoding: utf-8 -*-
"""
@File    : 数据预处理.py
@Time    : 2019-04-10 14:18
@Author  : final
@Email   : julyfinal@outlook.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np

# IRIS数据集由Fisher在1936年整理，包含4个特征
# （Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）），特征值都为正浮点数，单位为厘米。
# 目标值为鸢尾花的分类（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））。
from sklearn.datasets import load_iris

# 导入IRIS数据集
iris = load_iris()
# 特征矩阵
# print(iris.data)
# 目标向量
# print(iris.target)

# 无量纲化

# 标准化，返回值为标准化后的数据
from sklearn.preprocessing import StandardScaler
# StandardScaler().fit_transform(iris.data)

# 区间缩放，返回值为缩放到[0, 1]区间的数据
from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler().fit_transform(iris.data)

# 归一化，返回值为归一化后的数据 //说明：标准化是对列，归一化是对行
from sklearn.preprocessing import Normalizer
# Normalizer().fit_transform(iris.data)

# 二值化，阈值设置为3，返回值为二值化后的数据
from sklearn.preprocessing import Binarizer
# Binarizer(threshold=3).fit_transform(iris.data)

# 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(categories='auto', sparse=False)
# enc.fit([[0, 0, 3],
#          [1, 1, 0],
#          [0, 2, 1],
#          [1, 0, 2]])
#
# ans = enc.transform([[0, 1, 3]])  # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
# print(ans)  # 输出 [[ 1.  0.  0.  1.  0.  0.  0.  0.  1.]]

# 独热编码与标签编码组合实现
# 对上述补充
from sklearn.preprocessing import LabelEncoder
# setData=pd.DataFrame([['F'],['M'],['F'],['M']])
# enc=LabelEncoder()
# res=enc.fit_transform(setData[0])
# en=OneHotEncoder(categories='auto', sparse=False)
# res=en.fit_transform(res.reshape(4,1))


#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
from numpy import vstack,nan
from sklearn.preprocessing import Imputer
# Imputer().fit_transform(vstack((np.array([nan, nan, nan, nan]), iris.data)))

#多项式转换
#参数degree为度，默认值为2
from sklearn.preprocessing import PolynomialFeatures
# PolynomialFeatures().fit_transform(iris.data)

#自定义转换函数为对数函数的数据变换
#第一个参数是单变元函数
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
res=FunctionTransformer(log1p).fit_transform(iris.data)
print(res)