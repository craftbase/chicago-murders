import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import src.data.extractor.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("victim_info_2012_2017.csv")

def predict_logistic_regression(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    #get X variables
    X = df[['age','time','Cause__Stabbing','Cause__Assault','Cause__Auto Crash','Cause__Other','Cause__Shooting','Cause__Strangulation']]
    y = df['Race__White']
    print ("splitting test and training data")
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=101,test_size=0.3)
    lr = LogisticRegression()
    print ("Fitting the training data into the linear regression model")
    lr.fit(X_train,y_train)
    print ("Predicting values of test data")
    predictions = lr.predict(X_test)
    print ("Printing classification report")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


def predict_random_forest(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    # get X variables
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print ("splitting test and training data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
    rf = RandomForestClassifier(n_estimators=800)
    print ("Fitting the training data into the random forest")
    rf.fit(X_train, y_train)
    print ("Predicting values of test data")
    predictions = rf.predict(X_test)
    print ("Printing classification report")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

def predict_k_nearest_neighbours(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    # get X variables
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print ("splitting test and training data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print ("Fitting the training data into k-nearest neighbour")
    print ("Predicting values of test data")
    predictions = knn.predict(X_test)
    print ("Printing classification report")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def predict_using_neural_network(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print ("converting to matrices")
    X = X.as_matrix
    y = y.as_matrix
    D = 8
    M = 8
    K = 2
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    P = forward(X,W1,b1,W2,b2)
    P = np.argmax(P, axis=1)

    print (classification_rate(y,P))



predict_using_neural_network(df)