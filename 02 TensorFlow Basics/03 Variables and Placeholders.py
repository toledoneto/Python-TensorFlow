# Varibles:
# * pesos e bias da RN
# * precisam ser inicializadas

import tensorflow as tf


with tf.Session() as sess:
    my_tensor = tf.random_uniform((4, 4), 0, 1)

    # criando a variavel, porém ela n está inicializada
    my_var = tf.Variable(initial_value=my_tensor)

    # enfim inicializando as variáveis
    init = tf.global_variables_initializer()
    sess.run(init)

    # agora podemos rodar a var
    print(sess.run(my_var))

# Placeholders
# * iniciamente vazios
# * usados para dar entrada nos exemplos de treinamento
# * precisam de um dtype declarado e opcionalmente o shape da entrada

    ph = tf.placeholder(tf.float32)
