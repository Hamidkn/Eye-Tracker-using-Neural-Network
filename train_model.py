import cv2
import csv
import os
import keras.models
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


images = list()
labels = list()
    
def train():

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels))
        model = Sequential()
        model.add(Conv2D(32, input_shape=(64,64,1),strides=(2, 2), kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3),strides=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3),strides=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3),strides=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='relu'))

        model.compile(
            optimizer='rmsprop',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy']
        )

        model_info = model.fit(x_train, y_train, epochs=50)

        model.evaluate(x_test, y_test, verbose=2)
        with open('summarymodel.txt','w') as file:
            with redirect_stdout(file):
                model.summary()
        model.save('model/model_1.h5')
        plt.plot(model_info.history['accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train'],loc='upper left')
        plt.savefig('fig_accuarcy.png')
        plt.show()
        
        plt.plot(model_info.history['loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train'],loc='upper left')
        plt.savefig('fig_loss.png')
        plt.show()
        
def main():
    
    with open('dataset/pupil_in_eyes.txt', 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for index, row in enumerate(reader):
                image = cv2.imread(os.path.join('dataset', row[0]), cv2.IMREAD_GRAYSCALE)

                height, width = image.shape
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                image = np.reshape(image, (64, 64, 1))
                y_ratio = image.shape[0] / height
                x_ratio = image.shape[1] / width
                pupil = (round(int(row[1]) * x_ratio), round(int(row[2]) * y_ratio))

                images.append(image)
                labels.append(pupil)
    train()

if __name__ == '__main__':
    main()


