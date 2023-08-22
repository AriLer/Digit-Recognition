import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.models import Sequential, load_model

# pre-prossesing
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28))
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def show_mnist(x_train):
    plt.figure(figsize=(10,4.5))
    for i in range(24):  
        plt.subplot(3, 8, i+1)
        plt.imshow(x_train[i].reshape((28,28)),cmap=plt.cm.binary)
        plt.axis('off')
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
    plt.show()

def buildModel ():
    # build the model
    model = Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # training the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

def prep_img(src):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(28,28), interpolation=cv2.INTER_AREA)
    img = np.invert(np.array([img]))
    image, thresh_img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
    return thresh_img


def print_predictions():
    image_number = 1 
    images = []
    predictions = []
    while os.path.isfile(f"images/digit_{image_number}.png"):
        try:
            img = prep_img(f"images/digit_{image_number}.png")
            predictions.append(np.argmax(model.predict(img)))
            images.append(img[0])
        except: 
            print('error')
        finally:
            image_number +=1;

    plt.figure(figsize=(10,4.5))
    for i in range(len(images)):
        plt.subplot(2,7,i+1)
        plt.imshow(images[i].reshape((28,28)), cmap=plt.cm.binary)
        plt.title(predictions[i])
        
        plt.axis('off')
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
    plt.show()

# show_mnist(x_train)

# model = buildModel();
# model.save('handwritten-cnn.model')
