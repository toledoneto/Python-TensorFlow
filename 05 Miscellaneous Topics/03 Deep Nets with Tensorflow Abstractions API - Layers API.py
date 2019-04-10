from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.contrib.layers import fully_connected


wine_data = load_wine()

print(type(wine_data))  # <class 'sklearn.utils.Bunch'>

print(wine_data.keys())  # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print(wine_data['DESCR'])

feat_data = wine_data['data']
labels = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                    random_state=101)

scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

##############################################################
# Layers API

# ONE HOT ENCODED
onehot_y_train = pd.get_dummies(y_train).as_matrix()
one_hot_y_test = pd.get_dummies(y_test).as_matrix()

# params
num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate = 0.01

# TF.PH
X = tf.placeholder(tf.float32, shape=[None, num_feat])
y_true = tf.placeholder(tf.float32, shape=[None, 3])

# func ativ
actf = tf.nn.relu

# criando camadas
hidden1 = fully_connected(X, num_hidden1, activation_fn=actf)
hidden2 = fully_connected(hidden1, num_hidden2, activation_fn=actf)
output = fully_connected(hidden2, num_outputs)

# func de perda
# logits é praticamente nossa predição
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# tf sessão
init = tf.global_variables_initializer()

training_steps = 1000
with tf.Session() as sess:
    sess.run(init)

    for i in range(training_steps):
        sess.run(train, feed_dict={X: scaled_x_train, y_true: onehot_y_train})

    # Get Predictions
    logits = output.eval(feed_dict={X: scaled_x_test})

    preds = tf.argmax(logits, axis=1)

    results = preds.eval()

print(classification_report(results, y_test))
