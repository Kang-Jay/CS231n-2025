from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    # Initialize the loss and gradient to zero.优先设置值为0的损失和梯度.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient 计算损失和梯度
    # W.shape() = (3073, 10), X_dev.shape() = (500, 3073), X[i].shape() = (3  73,)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        # W -> s
        # 这里用.dot而不是@，主要是历史兼容性原因，.dot在老版本numpy中用于向量和矩阵乘法，@是Python 3.5+的新语法糖，两者在这里等价。
        # 严格来说，行向量shape应为(1, 3073)，列向量为(3073, 1)。但在numpy中，shape为(3073,)的一维数组既不是严格的行向量也不是列向量，只是一个一维向量。numpy在一维数组和二维行/列向量之间有区别，但在大多数运算中会自动按行向量处理X[i]。
        scores = X[i].dot(W)

        # ds/dW
        # 注意求导并非对矩阵求导, 而是对矩阵中的每个元素求导. dW只不过是这些元素的导数拼接起来的.

		
        # s -> p
        # compute the probabilities in numerically stable way
        # 这里减去最大值是为了避免数值溢出，因为指数函数在非常大或非常小的输入下会变得不稳定。
        scores -= np.max(scores)
        p = np.exp(scores)
        # 归一化. p是softmax模型预测出的归一化概率
        p /= p.sum()
        
        # p -> L
        # 计算损失: y[i]是真实标签，取-logp[y[i]]作为损失
        logp = np.log(p)
        loss -= logp[y[i]]  


        # 对于真实类别y[i]，dW[:,y[i]]+=X[i] * (p[y[i]]-1), 对其他所有类别j，dW[:,j] += X[i] * p[j]
        p[y[i]] -= 1
        for j in range(num_classes):
            dW[:, j] += X[i] * p[j] 
        # 或者dW += np.outer(X[i], p)
            

    # 这里的loss不是hinge loss，而是softmax损失（交叉熵损失）. hinge loss通常用于SVM（支持向量机），其形式为max(0, 1 - s_yi + s_j)，而softmax损失是-log(softmax概率)，即-log(exp(s_yi)/sum_j exp(s_j))
    # reg是正则化强度的超参数，用于控制正则项对总损失的影响，防止模型过拟合。
    # loss / num_train 是将累计的损失对训练样本数进行平均，得到每个样本的平均损失。由于每次输入的样本数是不同的, 所以平均是必须的.
    # reg * np.sum(W * W) 是L2正则化项，对权重W的平方和进行惩罚，鼓励权重较小，提升模型泛化能力。
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    return loss, dW 


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    # 计算Loss, X -> s -> p -> L
    s = X@W
    s = s - np.max(s, axis=1, keepdims=True)
    p = np.exp(s)
    # keepdims=True的作用是保持维度，使得p_sum的形状为(500,1)而不是(500,)
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.sum(np.log(p[np.arange(num_train), y]))
    loss = loss / num_train + reg * np.sum(W * W)
    # 计算梯度
    p[np.arange(num_train), y] -= 1
    dW = X.T @ p
    dW = dW / num_train + 2 * reg * W

    return loss, dW
