import numpy as np
import pandas as pd
import matplotlib
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import auc, roc_curve

def loadData():
    df = pd.read_csv('data.csv')
    print(df.head())

    return df


def preprocess(df):
    # convert data frame to 2D numpy array
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    # X = X.reshape(X.shape[0],-1)
    # y = y.reshape(y.shape[0],1)

    print(X.shape)
    print(y.shape)
    # normalization
    X = (X - X.mean())/X.std()

    # train test split
    X_train, X_test, y_train,  y_test = train_test_split(X, y, train_size = 0.9)

    return X_train, X_test, y_train, y_test

def featureSelection():
    pass


def visualization():
    pass

def MLmodel(X_train, y_train):
    # model = Sequential()
    # model.add(Dense(24, activation = 'relu'))
    # model.add(Dense(12))
    # model.compile(loss='softmax')

    model = LogisticRegression(multi_class= 'multinomial', solver = 'sag')

    model.fit(X_train, y_train)

    return model

def cross_validation(X_train, y_train):
    pass

def predict(X, model):
    y_pred = model.predict(X)
    print(y_pred)
    print(y_pred.shape)
    return y_pred

def evaluation(y_pred, y_test):
    pass
    # print(auc(y_pred, y_test))

def main():
    df = loadData()
    X_train, X_test, y_train, y_test = preprocess(df)
    model = MLmodel(X_train, y_train)
    y_pred = predict(X_test, model)
    evaluation(y_pred, y_test)


if __name__=="__main__":
    main()
