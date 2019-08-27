# import some libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import keras libraries

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
targer = [(i+5)/100 for i in range(100)]
data = np.array(data)
target = np.array(data)
# Split data to train test data
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(data,targer,test_size=.2,random_state=0)



# part 1 crete a model

# step 1 create a classification model
model = Sequential()

# step 2 add LSTM Layer
model.add(LSTM((1),return_sequences=True,batch_input_shape=(None,None,1)))

model.add(LSTM((1),return_sequences=False))

model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test))

plt.plot(history.history['loss'])
plt.show()