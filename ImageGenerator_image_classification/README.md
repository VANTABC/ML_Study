# 케라스의 ImageDataGenerator을 이용한 이미지 증식 및 분류
## keras 2.4.3
## scikit-learn 0.24.2
## cudatoolkit 11.3.1
## cudnn 8.2.1
## imgaug 0.4.0
## tensorflow 2.5.0
### 숫자 이미지를 증식 시킴
### 메모리 감소를 위해 단일채널(Gray)로 변경하여 증식 및 학습
### 
## Tensor Board 사용법
'''
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, Y_train, epochs=40, batch_size=32, validation_data=(X_valid, Y_valid),callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])
'''
