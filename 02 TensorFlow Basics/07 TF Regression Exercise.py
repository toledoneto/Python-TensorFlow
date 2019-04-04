# Regression Exercise
#
# California Housing Data
#
# This data set contains information about all the block groups in California from the 1990 Census.
# In this sample a block group on average includes 1425.5 individuals living in a geographically compact area.
#
# The task is to aproximate the median house value of each block from the values of the rest of the variables.
#
#  It has been obtained from the LIACC repository.
#  The original page where the data set can be found is: http://www.liaad.up.pt/~ltorgo/Regression/DataSets.html.
#
# The Features:
#
# * housingMedianAge: continuous.
# * totalRooms: continuous.
# * totalBedrooms: continuous.
# * population: continuous.
# * households: continuous.
# * medianIncome: continuous.
# * medianHouseValue: continuous.

import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Import the cal_housing_clean.csv file with pandas
data = pd.read_csv('cal_housing_clean.csv')

print(data.columns)
print(data.describe())
print(data.info())

y = data['medianHouseValue']
x = data.drop(columns='medianHouseValue', axis=1)

print(x.head())
print(y.head())

# Separate it into a training (70%) and testing set(30%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Use sklearn preprocessing to create a MinMaxScaler for the feature data. Fit this scaler only to the training data.
# Then use it to transform X_test and X_train. Then use the scaled X_test and X_train along with pd.Dataframe
# to re-create two dataframes of scaled data.
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = pd.DataFrame(data=scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(data=scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

# Create the necessary tf.feature_column objects for the estimator.
# They should all be trated as continuous numeric_columns.
age = tf.feature_column.numeric_column('housingMedianAge')
nrooms = tf.feature_column.numeric_column('totalRooms')
nbedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
household = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age, nrooms, nbedrooms, population, household, medianIncome]

# Create the input function for the estimator object.
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,
                                                 y=y_train,
                                                 batch_size=10,
                                                 num_epochs=1000,
                                                 shuffle=True)

# Create the estimator model. Use a DNNRegressor
model = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)

# Train the model for ~1,000 steps
model.train(input_fn=input_func, steps=25000)

# Create a prediction input function and then use the .predict method off your estimator model
# to create a list or predictions on your test data
predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=x_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)
print("Predições: " + str(list(predictions)))

# Calculate the RMSE
final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

mse = mean_squared_error(y_test, final_preds)**0.5
print("RSME: " + str(mse))
