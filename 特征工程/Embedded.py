# -*- encoding: utf-8 -*-
"""
@File    : Embedded.py
@Time    : 2019-04-11 10:33
@Author  : final
@Email   : julyfinal@outlook.com
@Software: PyCharm
"""

# 基于L1的特征选择 (L1-based feature selection)
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X)

# 对于SVM和逻辑回归，参数C控制稀疏性：C越小，被选中的特征越少。对于Lasso，参数alpha越大，被选中的特征越少。
from sklearn.feature_selection import SelectFromModel
#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
# SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)


# 基于树的特征选择 (Tree-based feature selection)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# clf = ExtraTreesClassifier()
# clf = clf.fit(X, y)
# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X)
# print(X.shape,X_new.shape)

#将特征选择过程融入pipeline (Feature selection as part of a pipeline)
from sklearn.pipeline import Pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)