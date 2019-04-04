import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# fazendo a RN do tipo y = mx + b, com b = 5 e com adição do ruído
b = 5
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

# colocando as duas colunas juntas
my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)

# plotando um pequeno pedaço
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')

# como 1mi pontos é muita coisa, criamos pedaços "batches" de dados. Não existe um num correto, depende do caso
batch_size = 8

# criando valores aleatórios para m e b
rand_result = np.random.rand(2).astype('float32')
print(rand_result)

m = tf.Variable(rand_result[0])
b = tf.Variable(rand_result[1])

# placeholders
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m*xph + b

# função de custo do erro
error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# inicializando var
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    batches = 1000

    for i in range(batches):

        # cria um índice random para alimentar a RN
        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

print(model_m)
print(model_b)

#resultados
y_hat = x_data * model_m + model_b

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(x_data, y_hat, 'r')

plt.show()
