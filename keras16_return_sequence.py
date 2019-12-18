from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x)
print(y)
print("x.shape", x.shape)  # (4,3)
print("y.shape", y.shape)  # (4,3)

x = x.reshape((x.shape[0], x.shape[1], 1))
print(x)
print(x.shape)

# 2. 모델구성
model = Sequential()
# input_shape(컬럼수, 컬럼을 몇개씩 잘라서 작업할 것인지)
# LSTM 은 None, 3, 1 로 shape 로 3차원 텐서를 입력값으로 받음
# input : None,3,1 --> out: None,3,10 / return_sequences 의 디폴트는 False : 디멘션을 맞춰줌.
# LSTM 에서 LSTM 모델로 값을 전달하는 경우는 return_sequence 로 디멘션을 유지해서 넘겨야 함
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True))
# model.add(LSTM(10, activation='relu', return_sequences=True))
# LSTM 은 3차원 텐서를 입력값으로 받아야 함. 위의 아웃풋은 10이므로 (None, 10)으로 2차원으로 넘어오므로 오류
# return_sequences 의 디폴트는 False
model.add(LSTM(3))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
'''
# 3. 실행
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='mse', patience=10, mode='auto')
# model.fit(x, y, epochs=100, verbose=2)
model.fit(x, y, epochs=80000, callbacks=[early_stopping])

# predict 용 데이터
x_input = array([25,35,45])  # 1,3, ????
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
'''