

#import some libraries

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Part 1 Create model Classification

classifier = Sequential()

# Step 1 Convolution layer
classifier.add(Convolution2D(32,6,6,input_shape=(64,64,3),activation='relu')) 

# Step 2 Max Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

# Add Second Convolution Layer
classifier.add(Convolution2D(32,6,6,activation='relu')) 
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Flatten())

# Add Images To ANN Layer 
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

# compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Part 2 data preprocessing import data 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=20,
        validation_data=test_set,
        validation_steps=2000)



