# -*- encoding: utf-8 -*-
'''
@file        :PLA.py
@time        :2020/01/21 18:01:53
@author      :caijiqhx
'''

import numpy as np


# 提取数据集
def getDataSet(filename):

    # 提取原始数据
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()
    length = len(dataSet)

    # 提取 X, Y
    # X[i, 0] 是 threshold
    X = np.zeros((length, 5))
    Y = np.zeros((length, 1))
    for i in range(length):
        data = dataSet[i].strip().split()
        X[i, 0] = 1.0
        X[i, 1] = np.float(data[0])
        X[i, 2] = np.float(data[1])
        X[i, 3] = np.float(data[2])
        X[i, 4] = np.float(data[3])
        Y[i, 0] = np.int(data[4])

    return X, Y


def sign(x, w):
    if np.dot(x, w)[0] > 0:
        return 1
    else:
        return -1


# 原始的 PLA 算法
# X(n+1, m), Y(m, 1) 训练集
# 循环遍历数据集，发现错误后修正并从下一个元素继续判断，知道划分成功。
def naivePLA(X, Y, w):
    cnt = 0
    length = len(X)
    print(w)
    while True:
        flag = True
        for i in range(length):
            if np.dot(X[i], w)[0] * Y[i, 0] <= 0:
                flag = False
                w += Y[i, 0] * np.matrix(X[i]).T
                print(i, end=" ")
                cnt += 1

        if flag == True:
            break

    return flag, cnt, w

filename = "hw1_15_train.dat"
X, Y = getDataSet(filename)
w0 = np.zeros((5, 1))
flag, cnt, w = naivePLA(X, Y, w0)

print(flag)
print(cnt)
print(w)