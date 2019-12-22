from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[1])
print(Y_test[1])
print(X_train.shape) # (60000,28,28)
print(X_test.shape)  # (10000,28,28)
print(Y_train.shape) # (60000,)
print(Y_test.shape) # (10000,)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

# cnn 모델에 넣기 위해 행,a,b,1 을 맞추기 위해 reshape를 수행
# 255로 나눈 이유 : min max 스케일러 사용. 스케일하면 훨씬 더 좋은 성능이 나옴. 각 픽셀값의 최대값 255임.
X_train = X_train.reshape(X_train.shape[0], 28*28,1).astype('float32') / 255  # (60000,28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28*28,1).astype('float32') / 255  # (10000,28,28,1)
print("X_train.shape :", X_train.shape)  # (60000,28*28,1)
print("X_test.shape :", X_test.shape)  # (60000,28*28,1)
'''
분류 : classification

원-핫 인코딩 : to_categorical 함수 사용
    단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 
    다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다. 이렇게 표현된 벡터를 원-핫 벡터(One-hot vector)
    원-핫 인코딩을 두 가지 과정으로 정리해보겠습니다.
    (1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
    (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여

'''
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_test[0])  #[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print(Y_train.shape)  # (60000,10)
print(Y_test.shape)  # (10000,10)

# 컨볼루션 신경망의 설정
model = Sequential()
# 인풋 shape : 28 x 28 x 1
model.add(LSTM(3, input_shape=(28*28,1), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()

# softmax 를 사용하는 경우 loss 는 categorical_crossentropy 를 사용함.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=200, verbose=1,
                    callbacks=[early_stopping_callback])  # checkpointer

# 테스트 정확도 출력
pred = model.predict(X_test, Y_test)
print(pred)
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))