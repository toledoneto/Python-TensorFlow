import tensorflow as tf


print(tf.__version__)

hello = tf.constant("Hello")
world = tf.constant("World")

# o tipo é um Tensor, uma especie de array unidimensional
print(type(hello))

# para executar um cód tensorflow, precisamos fazer dentro de uma sessão
with tf.Session() as sess:
    result = sess.run(hello+world)

print(result)

# constantes
a = tf.constant(10)
b = tf.constant(20)

# novamente o tipo é Tensor
print(type(a))

# atenção ao termo add, ele conta qtas vzs chamamos uma operação
print(a+b)  # Tensor("add_1:0", shape=(), dtype=int32)
print(a+b)  # Tensor("add_2:0", shape=(), dtype=int32)

with tf.Session() as sess:
    result = sess.run(a+b)

print(result)

const = tf.constant(10)
# craindo uma matrix 4x4 cheia de n 10
fill_mat = tf.fill((4, 4), 10)

# mat de zeros
myzeros = tf.zeros((4, 4))

# mat de um
myones = tf.ones((4, 4))

# random normal dist
myrand = tf.random_normal((4, 4), mean=1, stddev=1.0)

# random uniform dist
myrandu = tf.random_uniform((4, 4), minval=1, maxval=1)

my_ops = [const, fill_mat, myzeros, myones, myrand, myrandu]

# para uso em jupyter notebook, a sessão interativa pode ser melhor
sess = tf.InteractiveSession()  # desse momento em diante, tudo estará em uma sessão

for op in my_ops:
    print(sess.run(op))
    print('\n')

a = tf.constant([[1, 2], [3, 4]])
print(a.get_shape())

b = tf.constant([[10], [100]])

result = tf.matmul(a, b)

print(sess.run(result))


