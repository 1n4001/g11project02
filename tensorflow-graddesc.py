import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

def getData(filePath):
    data = np.genfromtxt(filePath, delimiter=',')
    x, y = np.array(data[:,0:-1], dtype=float), np.array(data[:,-1],dtype=int)
    return x,y

def normalize_duration(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return (x - mu)/sigma

def append_bias(x):
    n_training_samples, n_dim  = x.shape[0], x.shape[1]
    return np.reshape(np.c_[np.ones(n_training_samples),x],[n_training_samples,n_dim + 1])

def one_hot_encode(y):
    n = len(y)
    n_unique = len(np.unique(y))
    one_hot_encode = np.zeros((n,n_unique))
    one_hot_encode[np.arange(n), y] = 1
    return one_hot_encode

all_x,all_y = getData('data/all-flippedRegistrationStatus.csv')
all_x = normalize_duration(all_x)
all_x = append_bias(all_x)
all_y = one_hot_encode(all_y)


n_dim = all_x.shape[1]
rnd_indices = np.random.rand(len(all_x)) < 0.80

train_x = all_x[rnd_indices]
train_y = all_y[rnd_indices]
test_x = all_x[~rnd_indices]
test_y = all_y[~rnd_indices]

learning_rate = 0.1
training_epochs = 100

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,2])
W = tf.Variable(tf.ones([n_dim,2]),name='Weight')

init = tf.global_variables_initializer()

y_ = tf.nn.sigmoid(tf.matmul(X,W))
cost_function = tf.reduce_mean(tf.reduce_sum((-Y * tf.log(y_)) - ((1 - Y) * tf.log(1 - y_)), 
reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sess.run(optimizer,feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,sess.run(cost_function,
        	feed_dict={X: train_x,Y: train_y}))
    
    y_pred = sess.run(y_ , feed_dict={X: test_x})
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ",(sess.run(accuracy, feed_dict={X: test_x, Y: test_y})))
    
