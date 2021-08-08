from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras import regularizers
from keras import layers, models
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import os
from glob import glob
from PIL import Image
import numpy as np
import datetime


# PokemonData 폴더안의 폴더명들을 categories에 리스트로 할당
dataFolderDir = "C:/GitRepo/ML_Study/VGG16_image_classification/PokemonData/"
categories = os.listdir(dataFolderDir)


# 이미지 전처리 사이즈 할당
image_w = 224
image_h = 224
pixels = image_h * image_w * 3

X = []
Y = []


# 이미지를 불러와 리사이징
for idx, category in enumerate(categories):
    #one-hot encoding
    label = [0 for i in range(len(categories))]
    label[idx] = 1

    image_dir = dataFolderDir + category
    files = glob(image_dir+"/*.jpg")

    for file in files:
        img = Image.open(file)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        
        X.append(data)
        Y.append(label)



# train, val, test 데이터로 분리
X = np.array(X)
Y = np.array(Y)

s = np.arange(X.shape[0])
np.random.shuffle(s)

X, Y = X[s], Y[s]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=34)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, stratify=Y_train, random_state=34)


# 일반화
X_train = X_train.astype(float) / 255
X_valid = X_valid.astype(float) / 255
X_test = X_test.astype(float) / 255


# 모델 구축
input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')


##vgg16 모델 불러오기
pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
pre_trained_vgg.trainable = False
pre_trained_vgg.summary()


##vgg16 밑에 레이어 추가
additional_model = models.Sequential()
additional_model.add(pre_trained_vgg)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(150, activation='softmax'))

additional_model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['acc'])


#모델 훈련 및 텐서보드
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
additional_model.fit(X_train, Y_train, 
                    batch_size=32, 
                    epochs=1000, 
                    validation_data=(X_valid, Y_valid),
                    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])


# 평가 예측
loss, mse = additional_model.evaluate(X_test, Y_test, batch_size=32)
print('acc : ', mse)

y_predict = additional_model.predict_classes(X_test)
print(y_predict)