# import some libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encode label variable
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
lb1 = LabelEncoder()
X[:,1] = lb1.fit_transform(X[:,1])

lb2 = LabelEncoder()
X[:,2] = lb2.fit_transform(X[:,2])

oneHot = OneHotEncoder(categorical_features=[1])
X = oneHot.fit_transform(X).toarray()

# Remove Dummy Variable
X = X[:,1:]

# split independant variable to train test data
from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(X,y,test_size=.2)

# Add StanderScaller

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Create Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


# Add Hidden Layer
# init the initial weights
model.add(Dense(output_dim=20,activation='relu',input_dim=11,init='uniform'))
model.add(Dense(output_dim=20,activation='relu',init='uniform'))
model.add(Dense(output_dim=20,activation='relu',init='uniform'))
model.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=100)

# predect y
y_pred = model.predict(x_test)
y_pred = (y_pred > .5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_pred,y_test)
