import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


# criando um dataset irreal com sklearn com 100 amostras, 3 atributos e 2 centros
# data[0] são os dados em si
# data[1] são as labels que corresponde a cada input
data = make_blobs(n_samples=100, n_features=3, centers=2, random_state=101)

scaler = MinMaxScaler()

# estamos lidando com uma simulação de aprendizado não supervisionado, não temos motivo para seprar em treino e teste
scaled_data = scaler.fit_transform(data[0])

data_x = scaled_data[:, 0]
data_y = scaled_data[:, 1]
data_z = scaled_data[:, 2]

# plotando em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x, data_y, data_z, c=data[1])

plt.show()

############################################################################
# Linear Autoencoder
num_inputs = 3  # 3 dimensões de entrada
num_hidden = 2  # 2 dimensões para representar a entrada
num_outputs = num_inputs  # Must be true for an autoencoder!

learning_rate = 0.01

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

print(output_2d.shape)  # (100,2)

plt.scatter(output_2d[:, 0], output_2d[:, 1], c=data[1])

plt.show()
