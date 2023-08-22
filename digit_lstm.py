import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model


# Adding layers to model
# model = keras.Sequential()
# model.add(layers.LSTM(64, input_shape=(None, 28)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(10))
# print(model.summary())


# loading  MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_validate, y_validate = x_test[:-10], y_test[:-10]
x_test, y_test = x_test[-10:], y_test[-10:]
plt.figure(figsize=(15,4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(x_train[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

# # compiling model
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer="sgd",
#     metrics=["accuracy"],
# )


# # train and fit model
# model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1)
# model.fit(x_train, y_train, validation_data=(x_validate, y_validate), batch_size=64, epochs=10)


# model.save('handwritten-lstm.model')


model = load_model('handwritten-lstm.model');
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)

image_number = 1
plt.figure(figsize=(15,4.5))
while os.path.isfile(f"images/digit_{image_number}.png"):
    try:
        # prepare image
        img = cv2.imread(f"images/digit_{image_number}.png")[:,:,0]
        img = cv2.resize(src=img, dsize=(28,28), interpolation=cv2.INTER_AREA)
        img = np.invert(np.array([img]))
        img = cv2.bitwise_not(img)   
        img = tf.cast(tf.divide(img, 255) , tf.float64)              
        plt.imshow(img[0], cmap=plt.cm.binary)
        # plt.show()
        
        img = tf.image.adjust_contrast(img, 2.5)
        plt.imshow(img[0], cmap="gray")
        plt.imshow(x_train[i].reshape((28,28)),cmap=plt.cm.binary)
        plt.subplot(2, 7, image_number)
        plt.axis('off')
        prediction = model.predict(img)
        print(f'this digit is a {np.argmax(prediction)}')
    except: 
        print('error')
    finally:
        image_number +=1;
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

# image_number = 1
# while os.path.isfile(f"images/digit_{image_number}.png"):
#     try:
#         img = cv2.imread(f"images/digit_{image_number}.png")[:,:,0]
#         img = cv2.resize(src=img, dsize=(28,28), interpolation=cv2.INTER_AREA)
#         img = np.invert(np.array([img]))
#         # img = tf.image.adjust_contrast(img, 2.5)
#         prediction = model.predict(img)
#         print(f'this digit is a {np.argmax(prediction)}')
#         plt.imshow(img[0], cmap="gray")
#         plt.show()
#     except: 
#         print('error')
#     finally:
#         image_number +=1;
