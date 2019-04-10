# Linear Autoencoder for PCA - EXERCISE

# Follow instructions below to reduce a 30 dimensional data set for classification into a 2-dimensional dataset!
# Then use the color classes to see if you still kept the same level of class separation in the dimensionality reduction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

data = pd.read_csv('anonymized_data.csv')

print(data.head())
print(data.info())
print(len(data['EJWY']))  # (500, 0)

##################################################################
# Scale the Data
X = data.drop('Label', axis=1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

##################################################################
# Linear Autoencoder
num_inputs = 30
num_hidden = 2
num_outputs = num_inputs

learning_rate = 0.001

# TF PH
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# Camadas
hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

# func perda
loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# TF sessão
init = tf.global_variables_initializer()
num_steps = 1000

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_steps):
        sess.run(train, feed_dict={X: scaled_data})

    # separamos apenas a saída da camada interna
    output_2d = hidden.eval(feed_dict={X: scaled_data})

print(output_2d.shape)  # (500,2)

plt.scatter(output_2d[:, 0], output_2d[:, 1], c=data['Label'])

plt.show()

