import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from util import timer


def p1(W, b, X):
    return 1/(1+np.exp(-(np.dot(W.T, X)+b)))


def calLoss(W, b, validData):
    rowN, colN = validData.shape
    param = validData[:, 0:colN-1]
    result = validData[:, colN-1]
    Y = p1(W, b, param.T)
    loss = -1/rowN*np.sum(np.multiply(result, np.log(Y)) +
                          np.multiply(1-result, np.log(1-Y)))
    correntN = 0
    for i in range(rowN):
        tmp = 0 if Y[0, i] < 0.5 else 1
        if tmp == result[i]:
            correntN += 1
    return loss, correntN/rowN


@timer(info="train")
def trainLR(data, validData, speed, epsilon):
    rowN, colN = data.shape
    paramN = colN-1
    param = data[:, 0:paramN]
    result = data[:, paramN]
    W = np.random.rand(paramN, 1)
    b = np.random.rand(1, 1)
    lossList = []
    correntPList = []
    while True:
        for i in range(rowN):
            x = param[i, :].reshape(paramN, 1)
            y = result[i]
            p = p1(W, b, x)
            db = p-y
            dW = x*db
            b -= db*speed
            W -= dW*speed
        loss, correntP = calLoss(W, b, validData)
        lossList.append(loss)
        correntPList.append(correntP)
        if loss < epsilon:
            break
    return W, b, lossList, correntPList


@timer(info="test")
def testLR(data, W, b):
    rowN, colN = data.shape
    return calLoss(W, b, data)


def train(trainData, validData, speed, epsilon):
    df = pd.read_csv(trainData)
    data = df.values
    df = pd.read_csv(validData)
    validData = df.values
    W, b, loss, correntP = trainLR(data, validData, speed, epsilon)
    f = open("logisticRegression.pickle", "wb")
    pk.dump((W, b), f)
    f.close()
    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(loss)), loss)
    ax1.set_xlabel('iteration number')
    ax1.set_ylabel('cross entropy loss')
    plt.savefig('./LRloss.png')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(correntP)), correntP)
    ax2.set_xlabel('iteration number')
    ax2.set_ylabel('corrent rate')
    plt.savefig('./LRcorrentP.png')
    plt.show()


def test(testData):
    df = pd.read_csv(testData)
    data = df.values
    f = open("logisticRegression.pickle", "rb")
    W, b = pk.load(f)
    loss, correntP = testLR(data, W, b)
    print("correntP:{}, crossEntropyLoss:{}".format(correntP, loss))


train("./data/trainData.csv", "./data/validData.csv", 0.2, 0.1)
test("./data/testData.csv")
