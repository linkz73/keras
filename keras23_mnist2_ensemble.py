# x_train 60000, 28, 28  -> x1, x2 각 30000
# y_train 60000,   -> y1, y2 각 30000


from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[1])
print(Y_test[1])
print(X_train.shape) # (60000,28,28)
print(X_test.shape)  # (10000,28,28)
print(Y_train.shape) # (60000,)
print(Y_test.shape) # (10000,)

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

# cnn 모델에 넣기 위해 행,a,b,1 을 맞추기 위해 reshape를 수행
# 255로 나눈 이유 : min max 스케일러 사용. 스케일하면 훨씬 더 좋은 성능이 나옴. 각 픽셀값의 최대값 255임.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255  # (60000,28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255  # (10000,28,28,1)

from sklearn.model_selection import train_test_split
# split, test 는 행이 같은 항목을 묶어서 split 해야 함, X, Y는 60000으로 행이 동일함.
X1_train, X2_train, Y1_train, Y2_train = train_test_split(X_train, Y_train, random_state=3, test_size=0.5)
X1_test, X2_test, Y1_test, Y2_test = train_test_split(X_test, Y_test, random_state=3, test_size=0.5)

# X1_train = X_train[:30000, :, :, :]  # (30000,28,28,1)
# X2_train = X_train[30000:, :, :, :]  # (30000,28,28,1)
# X1_test = X_test[:5000, :, :, :]
# X2_test = X_test[5000:, :, :, :]


print("X1_train.shape :", X1_train.shape)  # (30000,28,28,1)
print("X2_train.shape :", X2_train.shape)  # (30000,28,28,1)
print("Y1_test.shape :", Y1_test.shape)  # (5000,)
print("Y2_test.shape :", Y2_test.shape)  # (5000,)

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
Y1_train = Y_train[:30000,]  # (30000,10)
Y2_train = Y_train[30000:,]  # (30000,10)
Y1_test = Y_test[:5000,]  # (5000,10)
Y2_test = Y_test[5000:,]  # (5000,10)

print(Y1_test[0])  #[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print("Y1_train.shape",Y1_train.shape)  # (30000,10)
print("Y2_train.shape",Y2_train.shape)  # (30000,10)
print("Y1_test.shape",Y1_test.shape)  # (5000,10)
print("Y2_test.shape",Y2_test.shape)  # (5000,10)

# 컨볼루션 신경망의 설정
input1 = Input(shape=(28,28,1))
conv = Conv2D(32, kernel_size=(3,3), activation='relu')(input1)
conv = Conv2D(64, (3,3), activation='relu')(conv)
conv = MaxPooling2D(pool_size=2)(conv)
conv = Dropout(0.25)(conv)
middle1 = Flatten()(conv)

input2 = Input(shape=(28,28,1))
conv = Conv2D(32, kernel_size=(3,3), activation='relu')(input2)
conv = Conv2D(64, (3,3), activation='relu')(conv)
conv = MaxPooling2D(pool_size=2)(conv)
conv = Dropout(0.25)(conv)
middle2 = Flatten()(conv)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])  # output 만 묶어 주면 됨.

output1 = Dense(128)(merge1)
output1 = Dropout(0.5)(output1)
output1 = Dense(10, activation='softmax')(output1)

output2 = Dense(128)(merge1)
output2 = Dropout(0.5)(output2)
output2 = Dense(10, activation='softmax')(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

# softmax 를 사용하는 경우 loss 는 categorical_crossentropy 를 사용함.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit([X1_train,X2_train], [Y1_train,Y2_train], validation_data=([X1_test,X2_test], [Y1_test,Y2_test]), 
                    epochs=2, batch_size=200, verbose=1, callbacks=[early_stopping_callback])  # checkpointer

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate([X1_test,X2_test], [Y1_test,Y2_test])[1]))