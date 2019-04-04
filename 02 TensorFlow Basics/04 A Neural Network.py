# Vamos analisar uma entrada linear simples

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


# placeholder vazios serão preenchidos posteriormente
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a+b
mul_op = a*b

with tf.Session() as sess:
    # para preencher um placeholder, passamos um dic chamado
    # feef_dict em que a chave é no mome o PH e o valor é a valor que queremos passar
    add_result = sess.run(add_op, feed_dict={a: 10, b: 20})
    print(add_result)

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

# RN
n_features = 10  # num de atributos
n_dense_neurons = 3  # num de neuronios

# passamos None pois não sabemos ao certo qual é o shape de entrada uma vez que
# os dados serão enviados em porções de diferente num de entradas, porém com
# mesmo num de atributos a serem avaliados
x = tf.placeholder(tf.float32, (None, n_features))

# os pesos serão inicializados aleatoriamente
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
# BIAS
b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x, W)

z = tf.add(xW, b)

a = tf.sigmoid(z)

# inicializando var
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})

# esses valores não estão resolvidos pela RN pq os erros n foram usados para reajustar os pesos
print(layer_out)

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

# fazendo a RN do tipo y = mx + b

# criando um cnj de dados aleatórios com um pouco de ruído
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.figure()
plt.plot(x_data, y_label, '*')

# criando valores aleatórios para m e b
rand_result = np.random.rand(2)
print(rand_result)

m = tf.Variable(rand_result[0])
b = tf.Variable(rand_result[1])

error = 0

for x, y in zip(x_data, y_label):

    # y previsto
    y_hat = m*x + b

    # função de custo: minimizar erros
    error += (y-y_hat)**2

# optimizar: minimizar o erro com gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# inicializando var
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    training_steps = 1

    for i in range(training_steps):

        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)

y_pred_plot = final_slope*x_test + final_intercept

plt.figure()
plt.title("Apenas um treino")
plt.plot(x_data, y_label, '*')
plt.plot(x_test, y_pred_plot)


with tf.Session() as sess:
    sess.run(init)

    training_steps = 100

    for i in range(training_steps):

        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)

y_pred_plot = final_slope*x_test + final_intercept

plt.figure()
plt.title("Cem treinos")
plt.plot(x_data, y_label, '*')
plt.plot(x_test, y_pred_plot)

plt.show()
