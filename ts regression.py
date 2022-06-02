# Importing Libraries & Helper Functions
#First of all, we will need to import some libraries and helper functions. This includes TensorFlow and some utility functions that I've written to save time.

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

%matplotlib inline
tf.logging.set_verbosity(tf.logging.ERROR)

print('Libraries imported.')

# Importing the Data
#The dataset is saved in a data.csv file. We will use pandas to take a look at some of the rows.

df = pd.read_csv('data.csv', names = column_names)
df.head()

#Check Missing Data
#It's a good practice to check if the data has any missing values. In real world data, this is quite common and must be taken care of before any data pre-processing or model training.

df.isna().sum()

dtype: int64

# Data Normalization
#We can make it easier for optimization algorithms to find minimas by normalizing the data before training a model.

df = df.iloc[:,1:]
df_norm = (df - df.mean())/df.std()
df_norm.head()

# Convert Label Value
#Because we are using normalized values for the labels, we will get the predictions back from a trained model in the same distribution. So, we need to convert the predicted values back to the original distribution if we want predicted prices.

y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)
print(convert_label_value(0.350088))

# Create Training and Test Sets
# Select Features
#Make sure to remove the column price from the list of features as it is the label and should not be used as a feature.

x = df_norm.iloc[:, :6]
x.head()

# Select Labels
y = df_norm.iloc[:,-1]
y.head()

# Feature and Label Values
#We will need to extract just the numeric values for the features and labels as the TensorFlow model will expect just numeric values as input.

x_arr = x.values
y_arr = y.values
print('Features array shape ', x_arr.shape)
print('labels array shape', y_arr.shape)

# Train and Test Split
#We will keep some part of the data aside as a test set. The model will not use this set during training and it will be used only for checking the performance of the model in trained and un-trained states. This way, we can make sure that we are going in the right direction with our model training.

x_train,x_test, y_train,y_test= train_test_split(x_arr, y_arr, test_size = 0.05,
                                                 random_state = 0)
print('Trainig set:',x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)

# Create the Model
#Let's write a function that returns an untrained model of a certain architecture.

def get_model():
    model = Sequential([
        Dense(10, input_shape = (6,), activation = 'relu'),
        Dense(20, activation = 'relu'), 
        Dense(5, activation = 'relu'),
        Dense(1)
    ])
    model.compile(
        loss='mse',
        optimizer = 'adam'
    
    )
    return model
get_model().summary()

#Model Training
#We can use an EarlyStopping callback from Keras to stop the model training if the validation loss stops decreasing for a few epochs.

es_cb = EarlyStopping(monitor = 'val_loss', patience = 5)

model = get_model()
preds_on_untrained = model.predict(x_test)
history = model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 100,
    callbacks = [es_cb]

)
# Plot Training and Validation Loss
#Let's use the plot_loss helper function to take a look training and validation loss.

plot_loss(history)

 #Predictions
# Plot Raw Predictions
#Let's use the compare_predictions helper function to compare predictions from the model when it was untrained and when it was trained.

preds_on_trained = model.predict (x_test)
compare_predictions(preds_on_untrained, preds_on_trained,y_test)

#Plot Price Predictions
#The plot for price predictions and raw predictions will look the same with just one difference: The x and y axis scale is changed.

price_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_trained = [convert_label_value(y) for y in preds_on_trained]
price_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_untrained, price_trained, price_test)

 