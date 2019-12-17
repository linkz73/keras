from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12], [20000,30000,40000], [30000,40000,50000], [40000,50000,60000],
            [55,65,75]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,4000])

# 14,3 데이터를 스케일링해서 13,3으로 잘라 트레이닝하고 끝에 하나는 테스트용도
# x, y 둘다 잘라줘야 함.
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x)  # fit 을 하면서 가중치가 생성됨.
x = scaler.transform(x)
# x = scaler.fit_transform(x)

x_train = x[:-1, :]
x_test = x[-1:, :]
y_train = y[:-1]
y_test = y[-1:]
print(x_train)
print(x_test)
print(y_train)
print(y_test)
print("x_train.shape", x_train.shape)  # (13,3)
print("x_test.shape", x_test.shape)  # (1,3)
print("y_train.shape", y_train.shape)  # (13,)
# x = x.reshape((x.shape[0], x.shape[1], 1))

# 2. 모델구성
model = Sequential()
# input_shape(컬럼수, 컬럼을 몇개씩 잘라서 작업할 것인지)
# model.add(LSTM(40, activation='relu', input_shape=(3,1)))
model.add(Dense(40, activation='relu', input_shape=(3,)))
model.add(Dense(30))   # activation 의 디폴트 값은 linear
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
# 0 ~ 1 로 MinMax 이용해 피팅된 상태
model.fit(x_train, y_train, epochs=4000, verbose=0)

# 평가, 예측

yhat = model.predict(x_test)
print(yhat)

# x_input = array([25,35,45])  # (3,)
# print("x_input shape", x_input.shape)
# x_input = x_input.reshape((1,3))
# print("x_input shape", x_input.shape)
# x_input = scaler.transform(x_input)  # 이전에 가중치 계산이 된 scaler 를 가져와 적용
# print(x_input)
# yhat = model.predict(x_input)
# print(yhat)
