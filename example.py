#This example belongs to the Keras repository and was slightly modified  
#to apply Eigenvalue Decay on two of the dense layers of the MLP model


#*******************************************************************
#Original comment reporting the performance before Eigenvalue Decay
#After Eigenvalue Decay the test accuracy reaches 98.74%
'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
#*******************************************************************

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

#Importing Eigenvalue Decay regularizer: 
from EigenvalueDecay import Regularizer
from EigenvalueDecay import EigenvalueRegularizer


batch_size = 128
nb_classes = 10
nb_epoch = 36

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,),W_regularizer=EigenvalueRegularizer(0.001))) #Applying Eigenvalue Decay with C=0.001
model.add(Activation('relu'))
model.add(Dropout(0.2))


#model.add(Dense(512)) #line from the original Keras example (uncomment this line and comment the next line to recover the original model)
model.add(Dense(512,W_regularizer=EigenvalueRegularizer(0.01))) #Applying Eigenvalue Decay with C=0.01
model.add(Activation('relu'))
model.add(Dropout(0.2))

#model.add(Dense(10)) #line from the original Keras example (uncomment this line and comment the next line to recover the original model)
model.add(Dense(10,W_regularizer=EigenvalueRegularizer(0.001))) #Applying Eigenvalue Decay with C=0.001
model.add(Activation('softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])