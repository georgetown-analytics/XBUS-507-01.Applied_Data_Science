import os 
import sys 

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

features  = [
    "Length",
    "Diameter",
    "Height",
    "Whole",
    "Shucked",
    "Viscera",
    "Shell",
    "Rings"
]

#import data
data = pd.read_csv('/Users/juliawang1/Desktop/datascience/abalone/abalone.csv')
data.head()

data = data.drop(['Sex'], axis=1)
data.head()

from sklearn.model_selection import train_test_split as tts 

target = 'Rings'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# OLS 
from sklearn.linear_model import LinearRegression 

model = LinearRegression() 
model.fit(X_train, y_train)

yhat = model.predict(X_test)

r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)

print("r2={:0.3f} MSE={:0.3f}".format(r2,me))