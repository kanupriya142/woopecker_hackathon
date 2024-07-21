import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import random
from sklearn.preprocessing import StandardScaler

def getData():
    df = pd.read_csv("flood.csv")
    return df


def forecast(data):
    x, x1 = random.randint(0,49999), random.randint(0,49999)
    model = load_model("model.h5")
    test = pd.concat([data.loc[x], data.loc[x1]], axis=1)
    tst = test.transpose()
    y = tst["FloodProbability"]
    tst = tst.drop(["FloodProbability"], axis=1)
    scaler = StandardScaler()
    x_sc = scaler.fit_transform(tst)
    pred = model.predict(x_sc)
    return pred[0]