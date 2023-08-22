# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along.

A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions.

This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model
![image](https://github.com/ShamRathan/basic-nn-model/assets/93587823/25a8f032-a6bd-4d06-a7b9-bb5511609711)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Program developed by: SHAM RATHAN S
Register Number: 212221230093
```
### Importing Modules:
```
from google.colab import auth
import gspread
from google.auth import default

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den

from tensorflow.keras.metrics import RootMeanSquaredError as rmse

import pandas as pd
import matplotlib.pyplot as plt
```
### Authenticate & Create Dataframe using Data in Sheets:
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('SomDocs DL-01').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
```
### Assign X and Y values:
```
x = df[["Input"]] .values
y = df[["Output"]].values
```
### Normalize the values & Split the data::
```
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)
```

### Create a Neural Network & Train it:
```
ai_brain = Seq([
    Den(9,activation = 'relu',input_shape=[1]),
    Den(16,activation = 'relu'),
    Den(1),
])

ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')

ai_brain.fit(x_train,y_train,epochs=1000)
ai_brain.fit(x_train,y_train,epochs=1000)
```

### Plot the Loss:
```
loss_plot = pd.DataFrame(ai_brain.history.history)
loss_plot.plot()
```

### Evaluate the model:
```
err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)
```
### Predict for some value:
```
x_n1 = [[9]]
x_n_n = scaler.transform(x_n1)
ai_brain.predict(x_n_n)
```

## Dataset Information
![image](https://github.com/ShamRathan/basic-nn-model/assets/93587823/8ad848aa-c96b-4de8-9061-70ef29c0e714)


## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/ShamRathan/basic-nn-model/assets/93587823/ae931de0-18ac-426a-b7fc-fb9040cc4e55)


### Test Data Root Mean Squared Error
![image](https://github.com/ShamRathan/basic-nn-model/assets/93587823/6059072b-cacb-4b9c-a718-3aee60f3b065)


### New Sample Data Prediction
![image](https://github.com/ShamRathan/basic-nn-model/assets/93587823/cd2e33c7-4da4-4419-ada3-c293218eab73)


## RESULT:
Thus to develop a neural network regression model for the dataset created is successfully executed.
