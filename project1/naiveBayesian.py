import pandas as pd
import numpy as np
import pickle as pk
import math
from util import timer


@timer(info="train")
def trainBayesian(data, intervals, resultN):
    rowN, colN = data.shape
    pY = np.zeros(resultN)
    paramN = len(intervals)
    assert(colN == paramN+1)
    pX = [None]*resultN
    intervals = np.array(intervals)
    partN = np.ceil(1/intervals)
    for i in range(resultN):
        pX[i] = [None]*paramN
        for j in range(paramN):
            pX[i][j] = np.zeros(int(partN[j]))
    for i in range(rowN):
        result = data[i][paramN]
        pY[int(result)] += 1
        for j in range(paramN):
            value = data[i][j]
            index = math.floor(value/intervals[j])
            if value == 1:
                index -= 1
            pX[int(result)][j][index] += 1
    for i in range(resultN):
        for j in range(paramN):
            pX[i][j] = (pX[i][j]+1)/(pY[i]+1)
    pY = (pY+1)/(rowN+resultN)
    return pX, pY


@timer(info="test")
def testBayesian(data, intervals, resultN, pX, pY):
    rowN, colN = data.shape
    paramN = colN-1
    correntN = 0
    for i in range(rowN):
        solution = data[i][paramN]
        p = [None]*resultN
        for k in range(resultN):
            p[k] = pY[k]
            for j in range(paramN):
                value = data[i][j]
                index = math.floor(value/intervals[j])
                if value == 1:
                    index -= 1
                p[k] *= pX[k][j][index]
        r = p.index(max(p))
        if(r == solution):
            correntN += 1
    return correntN/rowN


def train(trainData, intervals, resultN):
    df = pd.read_csv(trainData)
    data = df.values
    pX, pY = trainBayesian(data, intervals, resultN)
    f = open("bayesian.pickle", "wb")
    pk.dump((pX, pY), f)
    f.close()


def test(testData, intervals, resultN):
    df = pd.read_csv(testData)
    data = df.values
    f = open("bayesian.pickle", "rb")
    pX, pY = pk.load(f)
    f.close()
    correntP = testBayesian(data, intervals, resultN, pX, pY)
    print(correntP)


intervals = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.1, 0.1,
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5]

train("./data/trainData.csv", intervals, 2)
test("./data/testData.csv", intervals, 2)
