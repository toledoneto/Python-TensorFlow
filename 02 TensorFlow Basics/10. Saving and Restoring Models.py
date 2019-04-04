import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


np.random.seed(101)
tf.set_random_seed(101)

# Full Network Example
#
# Let's work on a regression example, we are trying to solve a very simple equation:
#
# y = mx + b
#
# y will be the y_labels and x is the x_data. We are trying to figure out the slope and the intercept for the line
# that best fits our data!

# Artifical Data (Some Made Up Regression Data)
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

print(x_data)

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
plt.plot(x_data, y_label, '*')

# Variables
np.random.rand(2)

m = tf.Variable(0.39)
b = tf.Variable(0.2)

# Cost Function
error = tf.reduce_mean(y_label - (m * x_data + b))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Initialize Variables
init = tf.global_variables_initializer()

# Saving The Model
saver = tf.train.Saver()

# Create Session and Run!
with tf.Session() as sess:
    sess.run(init)

    epochs = 100

    for i in range(epochs):
        sess.run(train)

    # Fetch Back Results
    final_slope, final_intercept = sess.run([m, b])

    # ONCE YOU ARE DONE
    # GO AHEAD AND SAVE IT!
    # Make sure to provide a directory for it to make or go to. May get errors otherwise
    # saver.save(sess,'models/my_first_model.ckpt')
    saver.save(sess, 'new_models/my_second_model.ckpt')

# Evaluate Results
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')

# Loading a Model
with tf.Session() as sess:
    # Restore the model
    saver.restore(sess, 'new_models/my_second_model.ckpt')

    # Fetch Back Results
    restored_slope, restored_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)
y_pred_plot = restored_slope * x_test + restored_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')

plt.show()
