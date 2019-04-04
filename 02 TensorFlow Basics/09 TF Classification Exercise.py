# We'll be working with some California Census Data,
# we'll be trying to use various features of an individual
# to predict what class of income they belogn in (>50k or <=50k).

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read in the census_data.csv data with pandas
census = pd.read_csv('census_data.csv')

print(census.head())
print(census.info())
print(census.describe())
print(census.columns)


# TensorFlow won't be able to understand strings as labels, you'll need to use pandas
# .apply() method to apply a custom function that converts them to 0s and 1s.
def zero_or_one(x):
    if x == ' <=50K':
        return 0
    else:
        return 1


census['income_bracket'] = census['income_bracket'].apply(zero_or_one)
print(census.head())
print(census.columns)

# Perform a Train Test Split on the Data
labels = census['income_bracket']
x_data = census.drop('income_bracket', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)

# Create the Feature Columns for tf.esitmator
# Features numéricas
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# Features categóricas
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Put all these variables into a single list with the variable name feat_cols
feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country,
             age, education_num, capital_gain, capital_loss, hours_per_week]

# Create Input Function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,batch_size=100,num_epochs=None,shuffle=True)

# Create your model with tf.estimator
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

# Train your model on the data, for at least 5000 steps
model.train(input_fn=input_func, steps=5000)

# Evaluation
# Create a prediction input function
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                              batch_size=len(X_test),
                                              shuffle=False)


predictions = list(model.predict(input_fn=pred_fn))
print(predictions)
print(predictions[0])

# Create a list of only the class_ids key values from the prediction list of dictionaries,
# these are the predictions you will use to compare against the real y_test values.
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(final_preds[:10])

# Import classification_report from sklearn.metrics and get a full report of your model's performance on the test data.
print(classification_report(y_test, final_preds))
