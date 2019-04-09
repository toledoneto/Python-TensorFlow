# Word2Vec

# Cod baseado no tutorial disponível pelo TensorFlow
# https://www.tensorflow.org/tutorials/word2vec
# Raw: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py


import collections
import math
import os
import errno
import random
import zipfile
import numpy as np
from collections import Counter
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt
from  sklearn.manifold import TSNE


# data
data_dir = "word2vec_data/words"
data_url = 'http://mattmahoney.net/dc/text8.zip'


# essa função irá encontrar o folder com os dados ou, caso ainda n tenhamos, irá fazer o download no site indicado
def fetch_words_data(url=data_url, words_data=data_dir):
    # cria a pasta caso n exista
    os.makedirs(words_data, exist_ok=True)

    # path para o zip
    zip_path = os.path.join(words_data, "words.zip")

    # caso n exista o zip, faz o download
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    # obtém os dados do zip
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])

    # cria uma lista com todas as palavras
    return data.decode("ascii").split()


words = fetch_words_data()

print(len(words))  # 17005207 -> num de palavras do dataset

# Counter retorna um dic em que a chave é cada palavra ÚNICA, ou seja, sem repetição, do dataset
# e o valor é o num de vzs que essa palavra apareceu


# pegamos apenas 'vocab_size' palavras mais comuns
def create_counts(vocab_size=50000):
    vocab = [] + Counter(words).most_common(vocab_size)

    # colocando em um array. O '_' siginifica que vamos jogar fora o num de vzs que essa palavra apareceu
    # ou seja, queremos apenas a palavra e n sua freq
    vocab = np.array([word for word, _ in vocab])

    dictionary = {word: code for code, word in enumerate(vocab)}
    data = np.array([dictionary.get(word, 0) for word in words])
    return data, vocab


vocab_size = 50000

# pode demorar
data, vocabulary = create_counts(vocab_size=vocab_size)

print(data[0])  # 5241
print((words[100], data[100]))  # ('interpretations', 4195)
print(vocabulary.shape)  # (50000,)
print(vocabulary[np.random.randint(0, 50000)])  # 'randi'


# direto da documentação do TF
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# constantes
data_index = 0
batch, labels = generate_batch(8, 2, 1)

print(batch)  # array([3083, 3083, 3083, 3083, 3083, 3083, 3083, 3083])
print(labels)  # array([[  11],
                       #[5241],
                       #[  11],
                       #[5241],
                       #[5241],
                       #[  11],
                       #[  11],
                       #[5241]])

# batch
batch_size = 128

# dimensão dos vetores
embedding_size = 150

# qtas palavras a esq e a dir deverão ser avaliadas, qto maior > tempo pra treinar
skip_window = 1

# num de reusos de um input para gerar um label
num_skips = 2

# valor aleatorio de palavras para avaliar similaridade
valid_size = 16

valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

num_sampled = 64

# Learning Rate
learning_rate = 0.01

vocabulary_size = 50000

#########################################################################
# TensorFlow Placeholders and Constants

tf.reset_default_graph()

# INPUT
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# PERDA
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
trainer = optimizer.minimize(loss)

# calculando a semelhança de cossenos entre as palavras
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# SESSÃO
init = tf.global_variables_initializer()

num_steps = 200001

with tf.Session() as sess:
    sess.run(init)
    average_loss = 0
    for step in range(num_steps):

        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        empty, loss_val = sess.run([trainer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()


def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


print(type(final_embeddings))  # numpy.ndarray
print(final_embeddings.shape)  # (50000, 150)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

plot_only = 2000
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

labels = [vocabulary[i] for i in range(plot_only)]

print(low_dim_embs.shape)  # (2000, 2)

plot_with_labels(low_dim_embs, labels)

plot_with_labels(low_dim_embs, labels)
plt.xlim(-10, 10)
plt.ylim(-10, 10)

np.save('trained_embeddings_200k_steps', final_embeddings)

plt.show()
