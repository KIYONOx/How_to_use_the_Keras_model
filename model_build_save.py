import sys
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# MNIST dataset load
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# before：28 x 28 :［two-dimensional］ array x 60,000
# after ：784 : [one-dimensional] array x 60,000（256 : 0 〜 1 normalized ）
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000, 784).astype('float32') / 255

# before：0 〜 9 number x 60,000
# after ：one-hot       x 60,000
#         - 0 : [1,0,0,0,0,0,0,0,0,0]
#         - 1 : [0,1,0,0,0,0,0,0,0,0]
#         ...
Y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(y_test, 10)

# Sequential
model = Sequential()

# hidden layer 1
# - node：512
# - input：784
# - activating function：relu
# - dropout：0.2
model.add(Dense(512, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# hidden layer 2
# - node：512
# - activating function：relu
# - dropout：0.2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# output layer
# - node：10
# - activating function：softmax
model.add(Dense(10))
model.add(Activation('softmax'))

# summary output
model.summary()

# learning setting
# - target function：categorical_crossentropy
# - optimized algorithm：rmsprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# training
# - batch size：128
# - repeat count：20
model.fit(X_train, Y_train,
          batch_size=128,
          nb_epoch=20,
          verbose=1,
          validation_data=(X_test, Y_test))

# evaluation
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

#model save
model_json_str = model.to_json()
open('mnist_mlp_model.json', 'w').write(model_json_str)
model.save_weights('mnist_mlp_weights.h5');

from keras.models import model_from_json

# model load
model = model_from_json(open('mnist_mlp_model.json').read())
# weights result load
model.load_weights('mnist_mlp_weights.h5')