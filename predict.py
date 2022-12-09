import tensorflow as tf
import pandas as pd
import os
import numpy as np
from keras.models import load_model

os.chdir(f'{os.getcwd()}/training')

results = pd.DataFrame()

'''
# define the loss function
def custom_loss(data, y_pred):
    y_true = data[:, 0]
    gd = data[:, 1:]

    error = tf.square(y_true - y_pred)
    # if the goal difference is small (e.g. equal to 1),
    # multiply the error by 0.5 to "reward" the model
    # for making predictions that are closer to 0.5
    if gd > 1:
        error += (y_pred - 1) ** 2
    elif gd < 1:
        error += (y_pred - 0) ** 2
    else:
        error += (2*abs(y_pred - 0.5)) ** 2
    return error
    #return tf.reduce_mean(error, axis=-1)  # Note the `axis=-1`
'''

#model = load_model('trained_model.h5', custom_objects={'custom_loss': custom_loss})

# Load the TensorFlow model
model = tf.keras.models.load_model('trained_model.h5')

# Load the data into a Pandas dataframe
#test_data/
df = pd.read_csv('train.csv')
true_values = df['result']

df = df.drop(['result', 'gd'], axis=1)
df_norm = (df - df.mean()) / df.std()

# Run the prediction
predictions = model.predict(df_norm)

#results['prediction'] = prediction
pd.DataFrame(predictions).to_csv("predictions.csv")
