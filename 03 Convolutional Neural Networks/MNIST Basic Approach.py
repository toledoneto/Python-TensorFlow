# MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


#################################################################################
# obtendo os dados
# baixa os arqvs necessarios
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# type <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
# print(type(mnist))

#################################################################################

# verificando os dados. NOTA: eles estão tds no formato "vetor achatado", ou seja
# shape = (num de img, vetor achadato)

# dados de treino
# print(mnist.train.images)  # vetor com as imgs
# print(mnist.train.images.shape)  # (55000, 784)

# dados de teste
# print(mnist.test.images)
# print(mnist.test.images.shape)  # (10000, 784)

# dados para validação
# print(mnist.validation)  # vetor com as imgs
# print(mnist.validation.images.shape)  # (5000, 784)

#################################################################################
# mostrando uma img
# 'desachatando' o vetor
# single_image = mnist.train.images[1].reshape(28, 28)
# plt.imshow(single_image, cmap='gist_gray')

# NOTA: os vet já estão normalizados entre 0 e 1
# print(single_image.min())  # 0
# print(single_image.max())  # 1
#
# plt.show()


#################################################################################
# abordagem com TF - Passos:
# 1. placeholders
# 2. variaveis
# 3. grafos
# 4. função de perda
# 5. otimizador
# 6. sessão TF


# 1. placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
# None é o batchzise sendo indicado como tamanho desconhecido
y_true = tf.placeholder(tf.float32, [None, 10])

# 2. variaveis
# vamos iniciar os pesos em zero, o que não é uma boa opção, mas apenas para fins didáticos
# tf.zeros([num de imgs, possíveis labels])
W = tf.Variable(tf.zeros([784, 10]))
# BIAS com o msm num de labels
b = tf.Variable(tf.zeros([10]))

# 3. grafos
# serão nossas predições futuras
y = tf.matmul(x, W) + b

# 4. função de perda
# irá reduzir labels=y_true de logits=y
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# 5. otimizador
optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optmizer.minimize(cross_entropy)

# 6. sessão TF
# iniciando as variaveis globais
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # treino por 1k steps
    for step in range(1000):
        # TF já tem um func de alimentar batches que retorna, infelizmente ela n está presente em tds os casos
        # batch_x os dados
        # batch_y a label daqueles dados
        batch_x, batch_y = mnist.train.next_batch(batch_size=100)

        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # avaliando o modelo
    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))  # retorna bool

    # aqui mudamos o bool pra 0 e 1
    acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))



