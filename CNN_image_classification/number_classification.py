import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime

from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, Convolution2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 경로 설정 및 폴더 명 리스트 생성
groups_folder_path = 'C:/GitRepo/ML_Study/CNN_image_classification'
num_folder_list = ["num0", "num1", "num2", "num3", "num4", "num5", "num6", "num7", "num8", "num9"]
predictpath = "C:/GitRepo/ML_Study/CNN_image_classification/predict/"

model = Sequential()
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

# inputs = Input(shape=(150,256,3))
# x = Conv2D(64,(1,1), activation="relu")(inputs)
# x = AveragePooling2D((2,2))(x)
# x = Conv2D(128,(1,1), activation='relu')(x)
# x = AveragePooling2D((2,2))(x)
# x = Flatten()(x)
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(512, activation="relu")(x)
# x = Dropout(0.5)(x)
# prediction = Dense(10, activation='softmax')(x)


# accuracy 증가를 위해 size설정
image_w = 150
image_h = 256

X = []
Y = []

X_test = []
Y_test = []

# 데이터 전처리
for l, num_folder in enumerate(num_folder_list):
    label = [0 for i in range(10)]
    label[l] = 1
    image_dir_train = num_folder + "/train/"

    dir = groups_folder_path+image_dir_train
    for top, dir, f in os.walk(dir):
        for filename in f:
            img = cv2.imread(groups_folder_path+image_dir_train+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)

    image_dir_test = num_folder + "/test/"
    dir = groups_folder_path+image_dir_train
    for top, dir, f in os.walk(dir):
        for filename in f:
            img = cv2.imread(groups_folder_path+image_dir_train+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X_test.append(img/256)
            Y_test.append(label)

predict_data = []
dir = predictpath
for top, dir, f in os.walk(dir):
        for filename in f:
            img = cv2.imread(predictpath+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            predict_data.append(img/256)    

X = np.array(X)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
predict_data = np.array(predict_data)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=34)

# 훈련
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

# 텐서 보드
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, Y_train, epochs=40, batch_size=32, validation_data=(X_valid, Y_valid),callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])

# 평가 예측
loss, mse = model.evaluate(X_test, Y_test, batch_size=32)
print('acc : ', mse)

y_predict = model.predict_classes(predict_data)
print(y_predict)
