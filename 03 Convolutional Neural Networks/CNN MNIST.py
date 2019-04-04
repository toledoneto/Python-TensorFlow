# MNIST com CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# funções de ajuda
# 1. init pesos
# 2. init BIAS
# 3. Conv 2D
# 4. Camada de conv
# 5. Camada Normal (total// conectada)


# 1. init pesos
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


# 2. init BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


# 3. Conv 2D
# x [batch, H, W, Channels
# W [filter H, filter W, Channels in, Channels out]
def conv2d(x, W):
    # strides = num de neuronios pulados a cada cnj de filtros
    # padding = margem add ao final da img para que assegure dados a serem observados
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 4. Pooling
# x [batch, H, W, Channels
def max_pool_2by2(x):
    # ksize = tamanho da janela para cada dimensão de entrada
    # strides = stride da janela de cada dimensão de entrada
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 4. Camada de conv
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# 5. Camada Normal (total// conectada)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


#################################################################################
# abordagem com TF - Passos:
# 1. placeholders
# 2. camadas
# 3. grafos
# 4. função de perda
# 5. otimizador
# 6. sessão TF

# construindo o modelo

# 1. placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])  # nosso input
y_true = tf.placeholder(tf.float32, [None, 10])

# 2. Camadas
# 'desachatando' o vetor
# [28, 28, 1] => [h, w, canais de cor]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# primeira camada conv
# [5, 5, 1, 32] => compara 32 features para cada pedaço 5x5 de 1 canal de cor
convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
convo_1_pooling = max_pool_2by2(convo_1)

# segunda camada conv
# agora temos 32/64 pq a entrada dessa camada é a saída da anterior e, novam//, queremos analisar 32 feat
convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

# achatando o array novamente
convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])

# primeira camada completa
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# dropout - evitar overfitting
# prob de ser mantido, ou seja, se não ser descartado
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

# 3. grafos
# serão nossas predições futuras
# 10 são as possíveis saídas
y_pred = normal_full_layer(full_one_dropout, 10)

# 4. função de perda
# irá reduzir labels=y_true de logits=y_pred
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


# 5. otimizador
optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optmizer.minimize(cross_entropy)

# 6. sessão TF
# iniciando as variaveis globais
init = tf.global_variables_initializer()

steps = 5000

with tf.Session() as sess:
    sess.run(init)

    # treino
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size=50)

        sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

        # a cada 100 steps vamos imprimir a acc
        if i%100 == 0:
            print("On Step: {}".format(i))
            print("Accuracy: ")

            # avaliando o modelo
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))  # retorna bool

            # aqui mudamos o bool pra 0 e 1
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0}))
            print('\n')
