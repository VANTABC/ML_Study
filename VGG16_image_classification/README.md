# VGG16 모델을 이용한 이미지 분류
## keras 2.4.3
## scikit-learn 0.24.2
## cudatoolkit 11.3.1
## cudnn 8.2.1
## tensorflow 2.5.0
### 3개의 DENSE 층
### SGD 최적화 함수
## Tensor Board 사용법
'''
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, Y_train, epochs=40, batch_size=32, validation_data=(X_valid, Y_valid),callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])
'''
