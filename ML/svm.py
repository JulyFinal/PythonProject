# -*- coding: utf-8 -*-
# @Time    : 2018-11-18 20:45
# @Author  : Narcissus
# @Email   : julyfinal@outlook.com
# @File    : svm.py
# @Software: PyCharm

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
label = iris.target
data = iris.data

dataSet = StandardScaler().fit_transform(data)
train_data, test_data, train_label, test_label = train_test_split(dataSet, label, test_size=0.3, random_state=42)

model = SVC()

param_grid = {'C': [1e-3, 1e-2, 1e-1, 0.3333, 1, 10, 100, 1000],
              'gamma': [0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

#网格搜索
# grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1, cv=5, scoring='f1_macro')
# grid_search.fit(train_data, train_label)

grid_search = RandomizedSearchCV(model, param_grid, n_jobs=-1, verbose=1, cv=5, scoring='accuracy', refit=True,
                                 random_state=1234234)
grid_search.fit(train_data, train_label)

best_parameters = grid_search.best_estimator_.get_params()
print(best_parameters)

print(grid_search.score(train_data, train_label))
print(grid_search.score(test_data, test_label))