import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime
import imgaug.augmenters as iaa

from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, Convolution2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential


# 데이터 로드 및 증식, npy저장 함수
def imgRead(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
    img_set = np.vstack([np.array(seq.augment_images([img for i in range(seq_num)])),
                     img.reshape(1, img.shape[0], img.shape[1])]) / 255.
    img_set = img_set.astype(np.float32)

    return img_set


# 모델 생성
model = Sequential()
model.add(Convolution2D(64, (1, 1), activation='relu',
                        input_shape=(100,100,1)))
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


# Augmentation 설정
seq = iaa.Sequential([
    iaa.Add((-40,40)),
    iaa.Multiply((0.5, 1.5)),
    iaa.GammaContrast((0.5, 2.0)),
    iaa.MotionBlur(k=15),
    iaa.CropAndPad(percent=(-0.25, 0.25)),
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.imgcorruptlike.GlassBlur(severity=2),
    iaa.imgcorruptlike.GaussianBlur(severity=2),
    iaa.imgcorruptlike.DefocusBlur(severity=2),
    iaa.Affine(scale=(0.5, 1.5))
])

# accuracy 증가를 위해 size설정
image_w, image_h, seq_num = 100, 100, 100


# 경로 설정 및 폴더 명 리스트 생성
groups_folder_path = 'C:/GitRepo/ML_Study/imgaug_image_classification/'
save_path = 'C:/GitRepo/ML_Study/imgaug_image_classification/NPY/'
num_folder_list = ["num0", "num1", "num2", "num3", "num4", "num5", "num6", "num7", "num8", "num9"]
X, Y, dir_list = [], [], []

for l, num_folder in enumerate(num_folder_list):

    label = [0 for i in range(10)]
    label[l] = 1

    img_path = groups_folder_path + num_folder
    for filename in os.listdir(img_path):
        dir_list.append(img_path+'/'+filename)
        Y.extend([label for _ in range(seq_num + 1)])


# 데이터 전처리
for image_dir in dir_list:
    X.extend(imgRead(image_dir))



# npy 저장
np.save(save_path + 'SEQ_X.npy', np.array(X))
np.save(save_path + 'SEQ_Y.npy', np.array(Y))


# 이미지 로드 데이터 제거
# del X, Y


# 학습 및 검증 데이터 로드(npy 파일)
X = np.load(save_path + 'SEQ_X.npy')
X = X.reshape(-1, X.shape[1], X.shape[2], 1)
Y = np.load(save_path + 'SEQ_Y.npy')
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=34)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, stratify=Y_train, random_state=34)


# 라벨로 이루어진 넘파이 데이터 프레임 Y를 원핫코딩 -> 정수형으로 변환
#argmax()함수를 사용 -> 제일 큰 원소의 인덱스 번호를 리턴해준다 , 라벨은 2차원 배열이므로 axis 파라미터를 1로 준다
Y_test_label = np.argmax(Y_test, axis=1)


# 훈련
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


# 텐서 보드
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_valid, Y_valid),callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])


# 평가 예측
loss, mse = model.evaluate(X_test, Y_test, batch_size=32)
print('acc : ', mse)

y_predict = model.predict_classes(X_test)
print(y_predict)


# 맞춘 개수 추출
cnt = 0
for idx, pd in enumerate(Y_test_label):
    if pd == y_predict[idx]:
        cnt += 1


#아래 방법으로도 맞춘 개수 추출이 가능하다
#c = np.isclose(Y_test_label, y_predict)
#c = len(c[c == True])


# 맞춘 개수를 바탕으로 정확도 추출
ac = (cnt/len(Y_test_label))*100
print("Accuracy: ", ac)