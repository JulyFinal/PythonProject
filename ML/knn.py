import numpy as np
import operator


def createDateSet():
    """ 创建数据集,array 创建数组
    array数组内依次是打斗次数, 接吻次数
    group小组, labels标签"""
    group = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ["爱情片", "爱情片", "爱情片", "动作片", "动作片", "动作片"]
    return group, labels


def knn(inputData, trainData, label, k) -> str:
    # distance = np.sum((np.tile(inputData, (trainData.shape[0], 1)) - trainData) ** 2, axis=1) ** 0.5

    distance = np.sqrt(np.sum(np.power(np.tile(inputData, (trainData.shape[0], 1)) - trainData, 2), axis=1))

    sortArgData = distance.argsort()
    dict = {}
    for i in range(k):
        voteLabel = label[sortArgData[i]]
    dict[voteLabel] = dict.get(voteLabel, 1) + 1
    res = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    return res[0][0]


if __name__ == '__main__':
    dataSet, label = createDateSet()
    classLabel = knn([18, 90], dataSet, label, 3)
    print(classLabel)