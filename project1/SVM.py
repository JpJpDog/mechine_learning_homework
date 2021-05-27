import pandas as pd
from sklearn.svm import SVC
import numpy as np
import pickle as pk
from util import timer


@timer(info="train")
def trainSVM(param, result):
    clf = SVC()
    clf.fit(param, result.ravel())
    SVC(C=1.0,  gamma='auto', kernel='rbf')
    return clf


@timer(info="test")
def testSVM(param, result, clt):
    predict = clt.predict(param)
    correntN = 0
    for i in range(predict.shape[0]):
        if predict[i] == result[i]:
            correntN += 1
    return correntN


def train(trainData):
    df = pd.read_csv(trainData)
    data = df.values
    rowN, colN = data.shape
    param = data[:, 0:colN-1]
    result = data[:, colN-1:colN]
    clf = trainSVM(param, result)
    f = open("SVM.pickle", "wb")
    pk.dump(clf, f)
    f.close()


def test(testData):
    df = pd.read_csv(testData)
    data = df.values
    rowN, colN = data.shape
    param = data[:, 0:colN-1]
    result = data[:, colN-1:colN]
    f = open("SVM.pickle", "rb")
    clt = pk.load(f)
    correntN = testSVM(param, result, clt)
    print(correntN/rowN)


train("./data/trainData.csv")
test("./data/testData.csv")
