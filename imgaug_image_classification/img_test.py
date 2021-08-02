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

save_path = 'C:/yonsei/tbell/work_01/ML_Study/tbell/work2/NPY/'

X = np.load(save_path + 'SEQ_X.npy')
X = X.reshape(-1, X.shape[1], X.shape[2], 1)
for img in X[500:520]:
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
