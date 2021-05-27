import pandas as pd
import os
fileName = "./archive/train.csv"

df = pd.read_csv(fileName, header=0)
df = df.sample(frac=1.0)
rowN, colN = df.shape

df["price_range"] = df["price_range"].map(lambda x: 0 if x <= 1 else 1)
df = (df-df.min())/(df.max()-df.min())
trainN = int(rowN*0.8)
validN = int(rowN*0.1)
trainDf, validDf, testDf = df.iloc[:trainN], df.iloc[trainN:(
    trainN + validN)], df.iloc[(trainN+validN):]

if not os.path.exists("data"):
    os.mkdir("data")
trainDf.to_csv("./data/trainData.csv", index=None)
validDf.to_csv("./data/validData.csv", index=None)
testDf.to_csv("./data/testData.csv", index=None)
