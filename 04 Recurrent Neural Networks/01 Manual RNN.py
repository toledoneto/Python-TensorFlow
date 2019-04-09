# RNN manual com 3 camadas de neuronios usando TF

import numpy as np
import tensorflow as tf


# constantes
num_inputs = 2
num_neurons = 3

# Placeholders
# temos que criar manualmente cada PH de cada instante de tempo
x0 = tf.placeholder(tf.float32, [None, num_inputs])
x1 = tf.placeholder(tf.float32, [None, num_inputs])

# variáveis
Wx = tf.Variable(tf.random_normal(shape=[num_inputs, num_neurons]))  # camada de pesos para a primeira entrada
Wy = tf.Variable(tf.random_normal(shape=[num_neurons, num_neurons]))  # camada de pesos para a primeira saída
b = tf.Variable(tf.zeros([1, num_neurons]))

# grafos
y0 = tf.tanh(tf.matmul(x0, Wx) + b)  # primeira saída funciona normal
# para a segunda saída, pegamos a primeia saída (t-1), mult pelos pesos respectivo
# e somamos com a entrada atual no tempo (t)
y1 = tf.tanh(tf.matmul(y0, Wy) + tf.matmul(x1, Wx) + b)

# init var
init = tf.global_variables_initializer()

# BATCH 0:       example1 , example2, example 3
x0_batch = np.array([[0, 1],  [2, 3],    [4, 5]])  # DATA AT TIMESTAMP = 0

# BATCH 0:          example1 ,   example2,   example 3
x1_batch = np.array([[100, 101], [102, 103],  [104, 105]])  # DATA AT TIMESTAMP = 1

with tf.Session() as sess:
    sess.run(init)

    y0_output_vals, y1_output_vals = sess.run([y0, y1], feed_dict={x0: x0_batch, x1: x1_batch})

print(y0_output_vals)  # saída em t
print(y1_output_vals)  # saída em t+1
