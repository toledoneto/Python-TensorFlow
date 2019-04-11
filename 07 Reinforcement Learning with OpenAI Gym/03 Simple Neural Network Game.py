# esse explo não aprende com os erros passados, apenas usa a probabilidade do lado que vai escolher

import tensorflow as tf
import gym
import numpy as np


################################################################
# variáveis
num_inputs = 4
num_hidden = 4

# a saída será apenas a % de chance de ir para esq
num_outputs = 1

# inicializando o modelo
initializer = tf.contrib.layers.variance_scaling_initializer()

# TF.PH
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# Camadas
hidden_layer_one = tf.layers.dense(X ,num_hidden,activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)

# camada de saída: chance de ir pra esq
output_layer = tf.layers.dense(hidden_layer_one, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

# a % de ir pra dir é = a 1-%esq
probabilties = tf.concat(axis=1, values=[output_layer, 1 - output_layer])

# pega uma ação aleatoriamente
action = tf.multinomial(probabilties, num_samples=1)

# inicia as var
init = tf.global_variables_initializer()

################################################################
# TF sessão
saver = tf.train.Saver()

epi = 50
step_limit = 500
avg_steps = []
env = gym.make("CartPole-v1")

with tf.Session() as sess:
    init.run()
    for i_episode in range(epi):  # rodamos 50x
        obs = env.reset()

        for step in range(step_limit):  # rodamos 500x
            # ----------------------------------------
            # # descomentar caso deseje ver o desenho
            # env.render()
            # ----------------------------------------
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                avg_steps.append(step)
                print('Done after {} steps'.format(step))
                break
print("After {} episodes the average cart steps before done was {}".format(epi, np.mean(avg_steps)))
env.close()

