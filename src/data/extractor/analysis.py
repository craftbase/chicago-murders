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
import tensorflow as tf

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
    Z = sigmoid(np.dot(X,W1) + b1)
    A = np.dot(Z,W2) + b2
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

#methods for backpropagation
def derivative_w2(Z, T, Y):
    der = np.dot(Z.T,T - Y)

    return der

def derivative_w1(X, Z, T, Y, W2):

    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    der1 = X.T.dot(dZ)

    return der1


def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)


def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def predict_using_backpropagation(df):

    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print("converting to matrices")
    X = X.values
    y = y.values
    print(type(X))

    D = 8
    M = 8
    K = 2

    N = len(y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, y[i]] = 1

    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, alpha=0.5)
    plt.show()

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 1e-6
    costs = []
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(y, P)
            print ("cost:", c, "classification_rate:", r)
            costs.append(c)

        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()

def predict_using_neural_network(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)
    X = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y = df['Race__White']
    print ("converting to matrices")
    X = X.values
    y = y.values
    print (type(X))
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

def predict_using_neural_network_tf(df):
    df = preprocessing.clean_data(df)
    df = preprocessing.get_dummies(df)

    X1 = df[['age', 'time', 'Cause__Stabbing', 'Cause__Assault', 'Cause__Auto Crash', 'Cause__Other', 'Cause__Shooting',
            'Cause__Strangulation']]
    y1 = df['Race__White']

    X1 = X1.values
    y1 = y1.values

    numClasses = 2
    hiddenUnits = 10
    inputSize = 8

    print(X1.shape[1])
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, X1.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.Variable(True, dtype=tf.bool)

    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(X, hiddenUnits, activation=None, kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    fc = tf.nn.relu(fc)

    #W1 = tf.Variable(tf.truncated_normal([inputSize, hiddenUnits], stddev=0.1))
    #B1 = tf.Variable(tf.constant(0.1), [hiddenUnits])
    #W2 = tf.Variable(tf.truncated_normal([hiddenUnits, numClasses], stddev=0.1))
    #B2 = tf.Variable(tf.constant(0.1), [numClasses])

    #hiddenLayerOutput = tf.matmul(X, W1) + B1
    #hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
    #finalOutput = tf.matmul(hiddenLayerOutput, W2) + B2
    #finalOutput = tf.nn.relu(finalOutput)

    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), X)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))
    #opt = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss)
    #correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0
        for e in range(100):
            for batch_x, batch_y in get_batch(X1, y1, 1):
                iteration += 1
                feed = {X: batch_x,
                        y: batch_y
                        }

                train_loss, _, train_acc = sess.run([cost,optimizer,accuracy], feed_dict=feed)

def get_batch(data_x, data_y, batch_size=1):
    batch_n = len(data_x) // batch_size
    for i in range(batch_n):
        batch_x = data_x[i * batch_size:(i + 1) * batch_size]
        batch_y = data_y[i * batch_size:(i + 1) * batch_size]

        yield batch_x, batch_y

predict_using_neural_network_tf(df)