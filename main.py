import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# pre-prossesing

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # train model
# model.fit(x_train, y_train, epochs=4)

# model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model');

loss, accuracy = model.evaluate(x_test, y_test)

print('loss: ', loss)
print('accuracy: ', accuracy)

image_number = 1
while os.path.isfile(f"images/digit_{image_number}.png"):
    try:
        img = cv2.imread(f"images/digit_{image_number}.png")[:,:,0]
        img = cv2.resize(src=img, dsize=(28,28), interpolation=cv2.INTER_AREA)
        img = np.invert(np.array([img]))
        img = tf.image.adjust_contrast(img, 2)
        prediction = model.predict(img)
        print(f'this digit is a {np.argmax(prediction)}')
        # plt.imshow(img[0], cmap=plt.cm.binary)
        plt.imshow(img[0])
        plt.show()
    except: 
        print('error')
    finally:
        image_number +=1;