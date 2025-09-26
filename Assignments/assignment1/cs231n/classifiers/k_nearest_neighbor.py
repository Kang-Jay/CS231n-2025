from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                result = X[i] - self.X_train[j]
                dists[i,j] = np.sqrt(np.sum(result**2))
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # X.shape = (500, 3072)
            # self.X_train.shape = (5000, 3072)
            # 计算X[i] - self.X_train的每一行, difMtxRow.shape = (5000, 3072)
            difMtxRow = X[i] - self.X_train
            # 计算每一个testImg与5000个trainImg的距离, 结果为(5000,)
            dists[i, :] = np.sqrt(np.sum(difMtxRow**2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # 求test(500, 3072)与train(5000, 3072)的距离, 即每一行数据的平方差, 加和后开根
        # 方法: testRow平方 - 2*testRow*trainRow + trainRow平方 = (testRow - trainRow)平方
        # 两个平方项由sum后的(500,)(,5000)两个列向量broadcast而成, 相乘项由矩阵乘法计算
        # 1. 平方项:
        # test(500, 3072)->(500,)->(500,1)
        X2 = np.sum(X**2, axis=1)
        X2 = X2.reshape(-1, 1)
        # train(5000, 3072)->(5000,)->(1,5000)
        X2Train = np.sum(self.X_train**2, axis=1)
        X2Train = X2Train.reshape(1, -1)
        # 2. 相乘项:
        # test(500, 3072) * train(3072, 5000) = (500, 5000)
        testPlusTrain = X @ self.X_train.T
        # 3. 整合:
        dists = np.sqrt(X2 + X2Train - 2 * testPlusTrain)

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
          
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            # np.argsort()函数, 目标是返回value从小到大排序后的索引列表sortedIndexList
            sortedIdx = np.argsort(dists[i, :])
            # numpy 切片: 要哪边哪边是:
            kIdx = sortedIdx[:k]
            # 通过numpy 索引数组的方式进行向量化取值
            closest_y = self.y_train[kIdx]
            # np.bincount()函数, 忽略索引, 计算列表中每个值出现的次数, 返回以值为索引的countList
            countList = np.bincount(closest_y)
            # np.argmax()函数, 目标是最大值的索引, 并返回索引idx
            y_pred[i] = np.argmax(countList)

        return y_pred
