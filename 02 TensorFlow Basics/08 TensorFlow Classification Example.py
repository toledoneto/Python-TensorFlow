import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


diabetes = pd.read_csv('pima-indians-diabetes.csv')

print(diabetes.head())
print(diabetes.info())
print(diabetes.describe())
print(diabetes.columns)

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
                'Insulin', 'BMI', 'Pedigree']

# normalizando com o MinMax
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print(diabetes.head())

# colunas numéricas
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# colunas categóricas - dois modos principais
# * vocabulary list - quando sabemos as possíveis categorias que podem sair
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])

# * hash bucket - quando não sabemos as possíveis categorias que podem sair
#   * o algoritmo se envarrega de tentar achar os possíveis grupos, só passamos qtos gpos achamos que haverá no max
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

# verificando a idade
diabetes['Age'].hist(bins=20)

# transformando a idade em coluna categórica, dividindo em faixas estárias com tf
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_buckets]

# fazendo o treinamento com sklearn
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)

# input func
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Classificador Linear
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

# treinando
model.train(input_fn=input_func, steps=1000)

# avaliando
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

results = model.evaluate(eval_input_func)
print(results)

# fazendo predições
pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

predictions = model.predict(pred_input_func)
print(list(predictions))

# Classificador com Dense Neural Network
# Em hidden_units, passamos qtos neuronios queremos em cada camada
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)

# Se treinarmod o modelo com
# dnn_model.train(input_fn=input_func, steps=1000)
# teremos um erro de passar coluna categórica para a DNN, vamos contorná-lo

embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_column, age_buckets]

# nova input func
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)

dnn_model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

results = dnn_model.evaluate(eval_input_func)
print(results)

plt.show()
