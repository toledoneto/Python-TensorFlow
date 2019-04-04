# CNN-Project-Exercise
#
# We'll be using the CIFAR-10 dataset, which is very famous dataset for image recognition!
#
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.
#
# The dataset is divided into five training batches and one test batch, each with 10000 images.
# The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain
# the remaining images in random order, but some training batches may contain more images from one class than another.
#
# Between them, the training batches contain exactly 5000 images from each class.
# Follow the Instructions in Bold, if you get stuck somewhere, view the solutions video!
# Most of the challenge with this project is actually dealing with the data and its dimensions,
# not from setting up the CNN itself!

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#################################################################################
# adquirindo os dados
# Put file path as a string here
CIFAR_DIR = r'C:\Users\netot\Desktop\cursos\TensorFlow\03 Convolutional Neural Networks\cifar-10-batches-py\\'


# Load the Data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


# os diferentes arqvs que serão abertos
dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

print(batch_meta)
print('\n')
print(data_batch1.keys())

# Loaded in this way, each of the batch files contains a dictionary with the following elements:
# * data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
#           The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
#           The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
#           of the first row of the image.
# * labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label
#             of the ith image in the array data.

# The dataset contains another file, called batches.meta. It too contains a Python dictionary object.
# It has the following entries:
# * label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array
#                  described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

# Put the code here that transforms the X array!
X = data_batch1[b"data"]
X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

plt.figure()
plt.imshow(X[0])

plt.figure()
plt.imshow(X[1])

plt.figure()
plt.imshow(X[4])

# plt.show()


#################################################################################
# Helper Functions for Dealing With Data
def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarHelper():

    def __init__(self):
        self.i = 0

        # Grabs a list of all the data batches for training
        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]

        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


# Before Your tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()

# During your session to grab the next batch use this line
# (Just like we did for mnist.train.next_batch)
# batch = ch.next_batch(100)

#################################################################################
# Creating the Model
# Create 2 placeholders, x and y_true
x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(dtype=tf.float32)

# Helper Functions
# 1. init_weights
# 2. init_bias
# 3. conv2d
# 4. max_pool_2by2
# 5. convolutional_layer
# 6. normal_full_layer


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


# Create the Layers
# Create a convolutional layer and a pooling layer as we did for MNIST.
convo_1 = convolutional_layer(x, shape=[5, 5, 3, 32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

# Now create a flattened layer
convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8*8*64])

# Create a new full layer using the normal_full_layer function and passing in your flattend convolutional 2
# layer with size=1024. (You could also choose to reduce this to something like 512)
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# Now create the dropout layer with tf.nn.dropout, remember to pass in your hold_prob placeholder
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

# Finally set the output to y_pred by passing in the dropout layer into the normal_full_layer function.
# The size should be 10 because of the 10 possible labels
y_pred = normal_full_layer(full_one_dropout, 10)

# Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Optmizer
optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optmizer.minimize(cross_entropy)

# Before Your tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()

# During your session to grab the next batch use this line
# (Just like we did for mnist.train.next_batch)
# batch = ch.next_batch(100)

# Create a variable to intialize all the global tf variables
init = tf.global_variables_initializer()

steps = 5000

with tf.Session() as sess:
    sess.run(init)

    # treino
    for i in range(steps):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0}))
            print('\n')
