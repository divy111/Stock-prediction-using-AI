# IMPORT THE LIBRARIES USED
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import time
import datetime as dt

# GET THE DATA FROM THE GOOGLE/YAHOO FINANCE
symbol = input("Enter the stock name you want to search for: ")
data_source='google'
df = web.get_data_yahoo(symbol, dt.datetime(2012, 1, 1), dt.datetime.today())
print(df)
df.head()

# GET NUMBER OF ROWS AND COLUMS IN THE DATASET
print(df.shape)

#VISUALISE THE CLOSING PRICE
plt.figure(figsize=(10,5))
plt.title('CLOSING PRICE')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize= 10)
plt.ylabel('CLOSING PRICE', fontsize= 10)
plt.show()

#CREATE A NEW DATAFRAME WITH ONLY CLOSE COLUMN
data = df.filter(['Close'])
#Convert the dataframe to a numoy array
dataset = data.values
# GET the data length to train the model
traning_data_len = math.ceil(len(dataset)* .8)
print(traning_data_len)

#SCALE THE DATA
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data =scaler.fit_transform(dataset)
# print(scaled_data)

#CREATE THE TRAINING DATASET
#CREATE THE SCALED TRAINING DATA SET
train_data = scaled_data[0:traning_data_len , :]
# SSPLIT THE DATA INTO X-TRAIN AND Y-TRAIN 
x_train =[]
y_train =[]
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

# CONVERT THE X_TRAIN AND Y_TRAIN TO NUMPY ARRAYS
x_train, y_train = np.array(x_train), np.array(y_train)

#RESHAPE THE DATA
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#BUILD THE LSTM MODEL
model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

# COMPILE THE MODEL
model.compile(optimizer = 'adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size = 1, epochs=1)

#Create the testing dataset
#Create a new array containing scaled values from the 
test_data = scaled_data[traning_data_len-60:, :]
#CREATE THE SETS X_TESTS AND Y_TESTS
x_test = []
y_test = dataset[traning_data_len:, :]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

#CONVERT THE DATA INTO A NUMPY ARRAY
x_test = np.array(x_test)

# #RESHAPE THE DATA 
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

#GET THE MODELS PREDICTED PRICE VALUES
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#GET THE ROOT MEAN SQUARED ERROR
rmse = np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)

#Plot the data
train = data[:traning_data_len]
valid = data[traning_data_len:]
valid['predictions'] = predictions
#visualise the data
plt.figure(figsize = (10,5))
plt.title('Model')
plt.xlabel('Date', fontsize =10)
plt.ylabel('Close Price)', fontsize = 10)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc = 'lower right')
plt.show()

#show the valid and predicted prices
print(valid)

#get the quote
stock_quote =  web.get_data_yahoo(symbol, dt.datetime(2012, 1, 1), dt.datetime.today())
#create a new dataframe
new_df = stock_quote.filter(['Close'])
#GET LAST 60 DAYS CLOSING PRICE
last_60 = new_df[-60:].values
#Scale the data values between 0 to 1
last_60_scaled = scaler.transform(last_60)
#create an empty list
X_test = []
#Append past 60 days
X_test.append(last_60_scaled)
#Conert to numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("THE PREDICTED VALUE FOR THE NEXT DAY IS", pred_price)

time.sleep(100)