import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import preprocessing
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
    print "splitting test and training data"
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=101,test_size=0.3)
    lr = LogisticRegression()
    print "Fitting the training data into the linear regression model"
    lr.fit(X_train,y_train)
    print "Predicting values of test data"
    predictions = lr.predict(X_test)
    print "Printing classification report"
    print(classification_report(y_test, predictions))


def predict_random_forest(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    # get X variables
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print "splitting test and training data"
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
    rf = RandomForestClassifier(n_estimators=800)
    print "Fitting the training data into the random forest"
    rf.fit(X_train, y_train)
    print "Predicting values of test data"
    predictions = rf.predict(X_test)
    print "Printing classification report"
    print(classification_report(y_test, predictions))

def predict_k_nearest_neighbours(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    # get X variables
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print "splitting test and training data"
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print "Fitting the training data into k-nearest neighbour"
    print "Predicting values of test data"
    predictions = knn.predict(X_test)
    print "Printing classification report"
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

predict_k_nearest_neighbours(df)