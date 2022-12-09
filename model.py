import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir(f'{os.getcwd()}/training')

# Load the data
data = pd.read_csv('train.csv')

features = data.drop(['result', 'gd'], axis=1)
#results = data[['result', 'gd']]
results = data['result']
#gd = data['gd']

norm_feat = (features - features.mean()) / features.std()

######################################
'''
train_data, test_data, train_results, test_results = train_test_split(norm_feat, results, test_size=0.3)

# Create the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(110,)),
    #tf.keras.layers.Dropout(0.25),
    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

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

# Compile the model
model.compile(optimizer='adam', loss=custom_loss, metrics='mse')

history = model.fit(train_data, train_results, epochs=150, batch_size=1) #verbose=0)

# Create subplots for loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot the loss over time
ax1.plot(history.history['loss'])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Plot the accuracy over time
ax2.plot(history.history['mse'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')

# Show the plots
plt.show()

# Evaluate the model on the test data
score = model.evaluate(test_data, test_results, batch_size=1, verbose=0)
print(score)

model.save('trained_model.h5')
'''

##############################################

#'''
# Split the data into training and test sets
train_data, test_data, train_results, test_results = train_test_split(norm_feat, results, test_size=0.25)

# Create the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(110,)),
    #tf.keras.layers.Dropout(0.25),
    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model on the training data
history = model.fit(train_data, train_results, epochs=50, batch_size=50) #verbose=0)

# Create subplots for loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot the loss over time
ax1.plot(history.history['loss'])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Plot the accuracy over time
ax2.plot(history.history['accuracy'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')

# Show the plots
plt.show()

# Evaluate the model on the test data
score = model.evaluate(test_data, test_results, verbose=0)
print(score)

# Save the trained model
model.save('trained_model.h5')
#'''
