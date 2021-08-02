import os, re, glob
from typing import Generator
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, Convolution2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 경로 설정 및 폴더 명 리스트 생성
predictpath = "C:/GitRepo/ML_Study/ImageGenerator_image_classification/predict/"
model = tf.keras.Sequential()
model.add(Convolution2D(64, (1, 1), activation='relu',
                        input_shape=(150,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (1, 1),  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense((10),activation = 'softmax'))

model.summary()

# 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1./255,
                                 rotation_range = 30,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/GitRepo//ML_Study//ImageGenerator_image_classification/train',
                                                 target_size = (150, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_set = test_datagen.flow_from_directory('C:/GitRepo/ML_Study/ImageGenerator_image_classification/val',
                                            target_size = (150, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')                                           

test_set = test_datagen.flow_from_directory('C:/GitRepo/ML_Study/ImageGenerator_image_classification/test',
                                            target_size = (150, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# 훈련
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

# 텐서 보드
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model.fit_generator(training_set,
                         epochs = 30,
                         validation_data = val_set,
                         callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])

# 평가 예측
loss, mse = model.evaluate_generator(test_set)
print('acc : ', mse)

output = model.predict_generator(test_set, steps=5)
print(test_set.class_indices)
print(output)
