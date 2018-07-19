# -*- coding:utf-8 -*-
"""
最小二乘法实现线性回归。

直接使用最小二乘法求解
by LeYuan
"""
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets


# Ordinary least square(普通的最小二乘法)
def ordinary_least_square(X, Y):
        """
         例如：给定数据集D={(X1,y1),(X2,y2),(X3,y3),...,(Xm,ym)}, m为样本大小，其中Xi=(xi1;xi2;...xid),d为特征个数，欲
         求模型f(Xi)=w*Xi+b , w=(w1;w2;...wd)
          为了方便计算，我们扩展 x0=1, W=(b;w1;w2;...;wd),则
        :param X:  X为mxd维度矩阵 matrix
        :param Y:  Y为mx1维度矩阵  matrix
        :return: 回归系数 W=(b;w1;w2;...;wd)
        """
        X0 = mat(ones((X.shape[0], 1)))
        X = hstack((X0, X))  # extend x0=1 for each sample
        xTx = X.T * X
        if linalg.det(xTx) == 0.0:           # 计算矩阵的行列式
            print("This matrix is singular, cannot do inverse")  # 奇异矩阵，不能求逆
        else:
            return xTx.I * (X.T*Y)        # 返回回归系数 W=(b;w1;w2;...;wd)  xTx.I表示矩阵的逆


def predict(X, W):
    """
    预测数据
    :param X: X为mxd维度矩阵 matrix
    :param W: 回归系数 W=(b;w1;w2;...;wd)
    :return: the predict matrix
    """
    X0 = mat(ones((X.shape[0], 1)))
    X = hstack((X0, X))  # extend x0=1 for each sample
    Y = X*W
    return Y


def errors_compute(X, W, Y):
    """
     compute the errors
    :param X: X为mxd维度矩阵 matrix
    :param W: 回归系数 W=(b;w1;w2;...;wd)
    :param Y:  Y为mx1维度矩阵  matrix ,the real value matrix
    :return: the errors
    """
    y_predict = predict(X, W)
    total_error = np.mean(multiply(y_predict-Y, y_predict-Y))
    return total_error


def test_regulation():
    # load the diabetes dataset
    diabetes = datasets.load_diabetes()
    # use multiple feature
    diabetes_X =diabetes.data[:, 0:5]

    # Split the data into training/testing sets
    diabetes_X_train = mat(diabetes_X[:-20])
    diabetes_X_test = mat(diabetes_X[-20:])

    # Split the targets into training/testing sets
    diabetes_y_train = mat(diabetes.target[:-20]).T
    diabetes_y_test = mat(diabetes.target[-20:]).T

    W = ordinary_least_square(diabetes_X_train, diabetes_y_train)

    # the coeffcients
    print('Coefficients: \n', W)
    # the mean squared error
    yerror = predict(diabetes_X_train, W)-diabetes_y_train
    print("Mean squared error: %.2f" % mean(multiply(yerror, yerror)))

    # plot outputs
    y_test_array = diabetes_y_test.A.reshape(diabetes_y_test.shape[0],)
    y_predict_martrix = predict(diabetes_X_test, W)
    y_predict_array = y_predict_martrix.A.reshape(y_predict_martrix.shape[0],)
    plt.scatter(y_test_array, y_predict_array)
    plt.plot(y_test_array, y_test_array)

    plt.xlabel('true value')
    plt.ylabel('predict value')
    plt.title('ordinary_least_square')
    plt.show()


if __name__ == "__main__":
    test_regulation()
