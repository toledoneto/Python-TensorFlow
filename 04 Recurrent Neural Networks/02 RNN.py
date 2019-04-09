import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData():

    def __init__(self, num_points, xmin, xmax):

        self.xmax = xmax
        self.xmin = xmin
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):

        # pega uma amostra aleatória como incial
        rand_start = np.random.rand(batch_size, 1)

        # coloca esse início numa timeseries (ts)
        # como a amostra inicial é aleatória, pode ter qquer valor, precisamos garantir que esteja
        # dentro da ts esperada
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))

        # batch ts no eixo x
        batch_ts = ts_start * np.arange(0.0, steps+1) * self.resolution

        # cria o equivalente em y do eixo x anterior
        y_batch = np.sin(batch_ts)

        # formatando a RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts

        else:
            # passamos o valor anterior e o próx
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


# craindo dados (250 ptos, de zero a 10)
ts_data = TimeSeriesData(250, 0, 10)
# visualizando
plt.figure()
plt.plot(ts_data.x_data, ts_data.y_true)

# Num de steps no batch (será usado no futuro para as predições)
num_time_steps = 30

# batch de 1 ponto
y1, y2, ts = ts_data.next_batch(batch_size=1, steps=num_time_steps, return_batch_ts=True)
plt.figure()
plt.plot(ts.flatten()[1:], y2.flatten(), '*')

# plotando em cima dos dados criados
plt.figure()
plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single Training Instance')
plt.legend()
plt.tight_layout()

# treinando
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps+1)

plt.figure()
plt.title("A training instance", fontsize=14)
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), "bo", markersize=15, alpha=0.5, label="instance")
plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), "ko", markersize=7, label="target")

tf.reset_default_graph()

# Just one feature, the time series
num_inputs = 1
# 100 neuron layer
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1
# learning rate, 0.0001 default
learning_rate = 0.0001
# how many iterations to go through (training steps)
num_train_iterations = 2000
# Size of the batch of data
batch_size = 1

# PH
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

###############################################################################
# camadas RNN
# OutputProjectionWrapper garante que teremos o num de saídas especificados em num_outputs
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

# outras possibilidades
# ############# OPÇ 1 ##############
# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)

# ############# OPÇ 2 ##############
# n_neurons = 100
# n_layers = 3
#
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#           for layer in range(n_layers)])

# ############# OPÇ 3 ##############
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)

# ############# OPÇ 4 ##############
# n_neurons = 100
# n_layers = 3
#
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
#           for layer in range(n_layers)])

###############################################################################
# célula dinâmica de RNN
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


###############################################################################
# Optimizer e loss
loss = tf.reduce_mean(tf.square(outputs - y))  # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

###############################################################################
# sessão
init = tf.global_variables_initializer()

# salvando o modelo
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    # Save Model for Later
    saver.save(sess, "model/rnn_time_series_model")

###############################################################################
# fazendo predições em t+1
with tf.Session() as sess:
    saver.restore(sess, "model/rnn_time_series_model")

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

plt.figure()
plt.title("Testing Model")
# Training Instance
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5, label="Training Instance")
# Target to Predict
plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label="target")
# Models Prediction
plt.plot(train_inst[1:], y_pred[0, :, 0], "r.", markersize=10, label="prediction")

plt.xlabel("Time")
plt.legend()
plt.tight_layout()

###############################################################################
# gerando novas sequencias
# Seq 01
with tf.Session() as sess:
    saver.restore(sess, "model/rnn_time_series_model")

    # SEED WITH ZEROS
    zero_seq_seed = [0. for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])

plt.figure()
plt.plot(ts_data.x_data, zero_seq_seed, "b-")
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

# Seq 02
with tf.Session() as sess:
    saver.restore(sess, "model/rnn_time_series_model")

    # SEED WITH Training Instance
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        training_instance.append(y_pred[0, -1, 0])

plt.figure()
plt.plot(ts_data.x_data, training_instance, "b-")
plt.plot(ts_data.x_data[:num_time_steps], training_instance[:num_time_steps], "r-", linewidth=3)
plt.xlabel("Time")

plt.show()
