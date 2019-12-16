# keras13_lstm4를 카피해서 x와 y 데이터를 2개씩으로 분리, 2개의 인풋, 2개의 아웃풋 모델인 ensemble 모델을 구현하시오.

from numpy import array
# from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
x1 = x[0:3]
x2 = x[10:]
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
y1 = y[0:3]
y2 = y[10:]
print(x1)
print(x2)
print(y1)
print(y2)
print("x1.shape", x1.shape)  # (3,3)
print("x2.shape", x2.shape)  # (3,3)
print("y1.shape", y1.shape)  # (3,)
print("y2.shape", y2.shape)  # (3,)

x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))
print(x1.shape)  # 3,3,1
print(x2.shape)  # 3,3,1
print(y1.shape)  # 3,
print(y2.shape)  # 3,

# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# 순차형 모델과 시퀀셜 모델의 차이 : 인풋과 아웃풋을 명시적으로 표현
input1 = Input(shape=(3,1))
layer = LSTM(40, activation='relu')(input1)
layer = Dense(10)(layer)
middle1 = Dense(1)(layer)

input2 = Input(shape=(3,1))
layer2 = LSTM(40, activation='relu')(input2)
layer2 = Dense(10)(layer2)
middle2 = Dense(1)(layer2)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])  # output 만 묶어 주면 됨.

output1 = Dense(10)(merge1)
output1 = Dense(1)(output1)

output2 = Dense(3)(merge1)
output2 = Dense(1)(output2)

# model.summary()
model = Model(inputs = [input1, input2], outputs = [output1, output2])

# 3. 실행
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
# model.fit(x, y, epochs=100, verbose=2)
model.fit([x1,x2], [y1,y2], epochs=80000, callbacks=[early_stopping])

# predict 용 데이터
x1_input = array([10,11,12])  # 1,3, ????
x2_input = array([25,35,45])  # 1,3, ????
x1_input = x1_input.reshape((1,3,1))
x2_input = x2_input.reshape((1,3,1))

yhat1,yhat2 = model.predict([x1_input,x2_input])
print(yhat1)
print(yhat2)
