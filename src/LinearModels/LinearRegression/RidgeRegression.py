"""
岭回归实现线性回归
by LeYuan
"""

from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets


def ridge_regression(X, Y, lam):
    """
    例如：给定数据集D={(X1,y1),(X2,y2),(X3,y3),...,(Xm,ym)}, m为样本大小，其中Xi=(xi1;xi2;...xid),d为特征个数，欲
         求模型f(Xi)=w*Xi+b , w=(w1;w2;...wd)
          为了方便计算，我们扩展 x0=1, W=(b;w1;w2;...;wd),则
    :param X: mxd矩阵
    :param Y:  mx1矩阵
    :param lam: 得到lambda参数
    :return: W=(b;w1;w2;...;wd),
    """
    X, Y = featurescaling(X, Y)

    # extend x0=1 for each exmple
    x0 = mat(ones((X.shape[0], 1)))
    X = hstack((x0, X))

    xTx = X.T * X

    # 产生对角矩阵
    I = eye(X.shape[1])
    I[0][0] = 0  # w0 has no punish factor
    denom = xTx + I * lam

    if linalg.det(denom) == 0:
        print("this matrix is singular,  cannot do inverse")
        return
    else:
        W = denom.I * X.T * Y
        return W


def featurescaling(X, Y):
    """
    feature scaling : Mean Nomalization ,即(x-mean(x))/(max-min)
    :param X: mxd矩阵
    :param Y: mx1矩阵
    :return: X ,Y
    """
    # feature scaling
    yMean = mean(Y, 0)
    Y = (Y - yMean)/(amax(Y, 0)-amin(Y, 0))
    xMean = mean(X, 0)  # calc mean the substract it off
    xMax = amax(X, 0)
    xMin = amin(X, 0)
    X = (X - xMean)/(xMax - xMin)
    return X, Y


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
    diabetes_X = diabetes.data[:, 0:5]

    # Split the data into training/testing sets
    diabetes_X_train = mat(diabetes_X[:-20])
    diabetes_X_test = mat(diabetes_X[-20:])

    # Split the targets into training/testing sets
    diabetes_y_train = mat(diabetes.target[:-20]).T
    diabetes_y_test = mat(diabetes.target[-20:]).T

    # print((diabetes_X_test, diabetes_y_test))
    # print(featurescaling(diabetes_X_test, diabetes_y_test))
    diabetes_X_test, diabetes_y_test = featurescaling(diabetes_X_test, diabetes_y_test)

    # set lam
    lam = exp(-4)

    W = ridge_regression(diabetes_X_train, diabetes_y_train, lam)

    # the coeffcients
    print('Coefficients: \n', W)

    # the mean squared error
    diabetes_X_train, diabetes_y_train = featurescaling(diabetes_X_train, diabetes_y_train)
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