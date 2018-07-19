# -*- coding:utf-8 -*-
"""
梯度下降法实现多元线性回归
方法思想参考: http://m.blog.csdn.net/article/details?id=51169525
by LeYuan
"""
from numpy import *
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def gradient_decent(feature, Y, W, a, tolerance, iterations_num):
    """
     给定数据矩阵feature,计算参数列表w=(w0;w1;w2;w3;...;wn),学得模型hw(x)=w0X0+w1X1+...+wnXn=w.T*X
    :param feature: 特征值矩阵  mxn  m个样例，n个特征
    :param Y: 标记向量         Y=(y1;y2;y3;...;ym)   mx1
    :param W: 初始化参数                                  mx1
    :param a: 步长
    :param tolerance:下界
    :return:w=(w0;w1;w2;w3;...;wn)                      (n+1)x1
    """
    # feature->D extend x0=1 for each sample
    x0 = mat(ones((feature.shape[0], 1)))
    D = hstack((x0, feature))     # mx(n+1)

    # feature scaling

    converged = False
    for i in range(iterations_num):
        y_predict = D * W
        # compute error
        errors = np.mean(multiply(y_predict-Y, y_predict-Y))
        # while we haven't reached the tolerance yet, update each feature's weight
        # print (y_predict-Y).T.shape
        derive = ((y_predict-Y).T * D) / float(D.shape[0])          # 1x(n+1)
        W = W - a * derive.T
    print("The iteration_num:", iterations_num, " The errors:", errors)
    return W


def predict(feature, W):
    """
    输入数据，预测结果
    :param feature: 特征值矩阵
    :param W: 参数向量
    :return: 预测的向量
    """
    # feature->D extend x0=1 for each sample
    x0 = mat(ones((feature.shape[0], 1)))
    D = hstack((x0, feature))  # mx(n+1)
    # predict
    y_predict = D * W
    return y_predict


def test_regulation():
    # load the diabetes dataset
    diabetes = datasets.load_diabetes()
    # use multiple feature
    diabetes_X = diabetes.data[:, 0:5]

    # Split the data into training/testing sets
    diabetes_X_train = mat(diabetes_X[:-20])
    diabetes_X_test = mat(diabetes_X[-20:])

    # Split the targets into training/testing sets
    diabetes_y_train = mat(diabetes.target[:-20]).T
    diabetes_y_test = mat(diabetes.target[-20:]).T

    # 初始化W =(0;0;0;...;0)
    W = np.zeros([diabetes_X_train.shape[1]+1, 1])
    a = 0.32
    tolerance = 2200
    iterations_num = 5000
    W = gradient_decent(diabetes_X_train, diabetes_y_train, W, a, tolerance, iterations_num)
    print("W=", W)
    # 测试结果
    y_predict_martrix = predict(diabetes_X_test, W)
    print("y_predict_martrix", y_predict_martrix)

    # plot
    y_test_array = diabetes_y_test.A.reshape(diabetes_y_test.shape[0], )
    y_predict_array = y_predict_martrix.A.reshape(y_predict_martrix.shape[0], )
    plt.scatter(y_test_array, y_predict_array)
    plt.plot(y_test_array, y_test_array)

    plt.xlabel('true value')
    plt.ylabel('predict value')
    plt.title('gradientdecent')
    plt.show()


if __name__ == "__main__":
    test_regulation()
