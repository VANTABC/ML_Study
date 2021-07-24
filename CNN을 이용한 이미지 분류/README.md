# CNN을 이용한 이미지 분류
## keras 2.4.3
## scikit-learn 0.24.2
## cudatoolkit 11.3.1
## cudnn 8.2.1
## tensorflow 2.5.0
### 숫자 이미지를 이용하여 Train data와 Validation data로서 학습을 시킨 후 Test data로 모델의 성능을 측정

## Tensor Board 사용법
'''
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, Y_train, epochs=40, batch_size=32, validation_data=(X_valid, Y_valid),callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])
'''
