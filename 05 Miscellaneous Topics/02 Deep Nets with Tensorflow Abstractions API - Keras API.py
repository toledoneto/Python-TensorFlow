from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers


wine = load_wine()

print(type(wine))  # <class 'sklearn.utils.Bunch'>

print(wine.keys())  # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print(wine['DESCR'])

feat_data = wine['data']
labels = wine['target']

X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)

scaler = MinMaxScaler()

scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

##############################################################
# Keras API

# criando o modelo
dnn_keras_model = models.Sequential()
dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))

# função de perdas
dnn_keras_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# treinando
dnn_keras_model.fit(scaled_x_train, y_train, epochs=50)

# predições
predictions = dnn_keras_model.predict_classes(scaled_x_test)

print(classification_report(predictions, y_test))
