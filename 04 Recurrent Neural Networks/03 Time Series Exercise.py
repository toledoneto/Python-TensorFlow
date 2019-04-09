# Data: Monthly milk production: pounds per cow. Jan 62 - Dec 75
# src: https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


milk = pd.read_csv('monthly-milk-production.csv', index_col='Month')

print(milk.columns)
print(milk.tail(13))
print(milk.info())

# Make the index a time series by using
milk.index = pd.to_datetime(milk.index)

milk.plot()

###############################################################################
# Train Test Split

# Let's attempt to predict a year's worth of data. (12 months or 12 steps into the future)
# Create a test train split using indexing (hint: use .head() or tail() or .iloc[]).
# We don't want a random train test split, we want to specify that the test set is the last 12 months of data
# is the test set, with everything before it is the training.
test = milk.tail(12)
train = milk.head(156)

###############################################################################
# Scale the Data
scaler = MinMaxScaler()

# Use sklearn.preprocessing to scale the data using the MinMaxScaler.
# Remember to only fit_transform on the training data, then transform the test data.
# You shouldn't fit on the test data as well, otherwise you are assuming you would know about future behavior!
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)


###############################################################################
# Batch Function

# We'll need a function that can feed batches of the training data.
# We'll need to do several things that are listed out as steps in the comments of the function.
# Remember to reference the previous batch method from the lecture for hints.
def next_batch(training_data, batch_size, steps):
    """
    INPUT: Data, Batch Size, Time Steps per batch
    OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
    """

    # STEP 1: Use np.random.randint to set a random starting point index for the batch.
    # Remember that each batch needs have the same number of steps in it.
    # This means you should limit the starting point to len(data)-steps
    rand_start = np.random.randint(0, len(training_data) - steps)

    # STEP 2: Now that you have a starting index you'll need to index the data from
    # the random start to random start + steps + 1. Then reshape this data to be (1,steps+1)
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1, steps + 1)

    # STEP 3: Return the batches. You'll have two batches to return y[:,:-1] and y[:,1:]
    # You'll need to reshape these into tensors for the RNN to .reshape(-1,steps,1)
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

###############################################################################
# Setting Up The RNN Model

# The Constants
# Define the constants. You'll need the following:
#
#     Number of Inputs (1)
#     Number of Time Steps (12)
#     Number of Neurons per Layer (100)
#     Number of Outputs (1)
#     Learning Rate (0.03)
#     Number of Iterations for Training (4000)
#     Batch Size (1)
# a série temporal é a única entrada
num_inputs = 1
# num de epocas em cada batch de série
num_time_steps = 12
# num de neuronios
num_neurons = 100
# apenas uma saída
num_outputs = 1

# tx de aprendizado
learning_rate = 0.03
# num de iteracoes
num_train_iterations = 4000
# tamanho do batch
batch_size = 1

# Create Placeholders for X and y. (You can change the variable names if you want).
# The shape for these placeholders should be
# [None,num_time_steps-1,num_inputs]
# and
# [None, num_time_steps-1, num_outputs]
# The reason we use num_time_steps-1 is because each of these will be one step shorter than the original
# time steps size, because we are training the RNN network to predict one point into the
# future based on the input sequence.
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# create the RNN Layer
# -----------------------------------------------------------------------------------------
# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)
# -----------------------------------------------------------------------------------------
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)
# -----------------------------------------------------------------------------------------
# n_neurons = 100
# n_layers = 3
#
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#           for layer in range(n_layers)])
# -----------------------------------------------------------------------------------------
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
# -----------------------------------------------------------------------------------------
# n_neurons = 100
# n_layers = 3
#
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
#           for layer in range(n_layers)])
# -----------------------------------------------------------------------------------------

# Now pass in the cells variable into tf.nn.dynamic_rnn, along with your first placeholder (X)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

###############################################################################
# Loss Function and Optimizer

# Create a Mean Squared Error Loss Function and use it to minimize an AdamOptimizer,
# remember to pass in your learning rate
loss = tf.reduce_mean(tf.square(outputs - y))  # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Initialize the global variables
init = tf.global_variables_initializer()

# Create an instance of tf.train.Saver()
saver = tf.train.Saver()

###############################################################################
# Session

# Run a tf.Session that trains on the batches created by your next_batch function.
# Also add an a loss evaluation for every 100 training iterations.
# Remember to save your model after you are done training
with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = next_batch(scaled_train, batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    # Save Model for Later
    saver.save(sess, "./ex_time_series_model")

###############################################################################
# Predicting Future (Test Data)

# Show the test_set (the last 12 months of your original complete data set)
print(test)

# Now we want to attempt to predict these 12 months of data, using only the training data we had.
# To do this we will feed in a seed training_instance of the last 12 months of the training_set of data
# to predict 12 months into the future. Then we will be able to compare our generated 12 months to our actual
# true historical values from the test set!

###############################################################################
# Generative Session

# NOTE: Recall that our model is really only trained to predict 1 time step ahead,
# asking it to generate 12 steps is a big ask, and technically not what it was trained to do!
# Think of this more as generating new values based off some previous pattern,
# rather than trying to directly predict the future. You would need to go back to the original model and
# train the model to predict 12 time steps ahead to really get a higher accuracy on the test data.
# (Which has its limits due to the smaller size of our data set)

# Fill out the session code below to generate 12 months of data based off the last 12
# months of data from the training set. The hardest part about this is adjusting the arrays with their shapes and sizes
with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./ex_time_series_model")

    # Create a numpy array for your genreative seed from the last 12 months of the
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(scaled_train[-12:])

    # Now create a for loop that
    for iteration in range(12):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])

# Show the result of the predictions
print(train_seed)

# Grab the portion of the results that are the generated values and apply inverse_transform
# on them to turn them back into milk production value units (lbs per cow). Also reshape the results to be
# (12,1) so we can easily add them to the test_set dataframe
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12, 1))

# Create a new column on the test_set called "Generated" and set it equal to the generated results.
# You may get a warning about this, feel free to ignore it
test['Generated'] = results

test.plot()

plt.show()
