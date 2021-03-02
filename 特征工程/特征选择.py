# -*- encoding: utf-8 -*-
"""
@File    : 特征选择.py
@Time    : 2019-04-10 16:17
@Author  : final
@Email   : julyfinal@outlook.com
@Software: PyCharm
"""

# 当数据预处理完成后，我们需要选择有意义的特征输入机器学习的算法和模型进行训练。通常来说，从两个方面考虑来选择特征：
# 1、特征是否发散：如果一个特征不发散，例如方差接近于 0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。
# 2、特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优选选择。除移除低方差法外，本文介绍的其他方法均从相关性考虑。

from sklearn.datasets import load_iris
# 导入IRIS数据集
iris = load_iris()

# 根据特征选择的形式又可以将特征选择方法分为 3 种：
# Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
# Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
# Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

# 特征选择主要有两个目的：
# - 减少特征数量、降维，使模型泛化能力更强，减少过拟合；
# - 增强对特征和特征值之间的理解。

# Filter
#方差选择法，返回值为特征选择后的数据
# 假设某特征的特征值只有 0 和 1，并且在所有输入样本中，95% 的实例的该特征取值都是1，那就可以认为这个特征作用不大。
# 如果100% 都是 1，那这个特征就没意义了。当特征值都是离散型变量的时候这种方法才能用，如果是连续型变量，就需要将连续变量离散化之后才能用。
# 而且实际当中，一般不太会有 95% 以上都取某个值的特征存在，所以这种方法虽然简单但是不太好用。
# 可以把它作为特征选择的预处理，先去掉那些取值变化小的特征，然后再从接下来提到的的特征选择方法中选择合适的进行进一步的特征选择。

# 移除低方差的特征
#参数threshold为方差的阈值
from sklearn.feature_selection import VarianceThreshold
# VarianceThreshold(threshold=3).fit_transform(iris.data)
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))

# 单变量特征选择 　单变量特征选择的原理是分别单独的计算每个变量的某个统计指标，根据该指标来判断哪些指标重要，剔除那些不重要的指标。
# 　对于分类问题(y离散)，可采用：卡方检验，f_classif, mutual_info_classif，互信息
# 　对于回归问题(y连续)，可采用：皮尔森相关系数，f_regression, mutual_info_regression，最大信息系数
# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
from sklearn.feature_selection import SelectKBest,chi2
from scipy.stats import pearsonr
from numpy import array
from minepy import MINE
# SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target) # 卡方
# SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:pearsonr(x, Y), X.T))).T)), k=2).fit_transform(iris.data, iris.target) # 皮尔森相关系数

#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
#选择K个最好的特征，返回特征选择后的数据
# SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:mic(x, Y), X.T))).T)),k=2).fit_transform(iris.data, iris.target) #MIC指标

# Wrapper
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

# Embedded
from sklearn.feature_selection import SelectFromModel
#带L1惩罚项的逻辑回归作为基模型的特征选择
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
# SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)

#GBDT作为基模型的特征选择
from sklearn.ensemble import GradientBoostingClassifier
# SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)


