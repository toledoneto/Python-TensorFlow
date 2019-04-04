import tensorflow as tf

# criamos dois grafos
n1 = tf.constant(1)
n2 = tf.constant(3)

# criamos um 3 grafo com a operação
n3 = n1+n2

with tf.Session() as sess:
    result = sess.run(n1+n2)

print(result)

# ao criarmos uma app tf, criamos um novo grafo como default
# vamos busca-lo
print(tf.get_default_graph())  # endereço da memória do grafo defautl

# agora vamos criar uma novo grafo
g = tf.Graph()
print(g)  # endereço da memória do grafo criado

# setando um grafo como default
graph_one = tf.get_default_graph()

# e dps trocando por outro default
graph_two = tf.Graph()

# fazemos a troca na sessão
with graph_two.as_default():
    print(graph_two is tf.get_default_graph())

# aqui o graph_one volta a ser default
print(graph_two is tf.get_default_graph())
