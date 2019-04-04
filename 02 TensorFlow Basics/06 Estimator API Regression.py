import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# fazendo a RN do tipo y = mx + b, com b = 5 e com adição do ruído
b = 5
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

# colocando as duas colunas juntas
my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']), pd.DataFrame(data=y_true,columns=['Y'])],axis=1)

# plotando um pequeno pedaço
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

# estimator API
# geralmente, seguimos os passos:
# 1. definimos uma lista de colunas de atributos
# 2. criamos um Estimator Model
# 3. criamos uma função de input de dados
# 4. treinamos, avaliamos e predizemos usando o obj Estimator

# PASSO 1
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]

# PASSO 2
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# PASSO 4.1
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

# PASSO 3
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

# PASSO 4.2
estimator.train(input_fn=input_func, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

print("train metrics: {}".format(train_metrics))
print("eval metrics: {}".format(eval_metrics))

input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': np.linspace(0, 10, 10)}, shuffle=False)
print(list(estimator.predict(input_fn=input_fn_predict)))

predictions = []  # np.array([])
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(np.linspace(0, 10, 10), predictions, 'r')

plt.show()
