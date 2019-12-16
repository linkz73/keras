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
model.add(LSTM(40, activation='relu', input_shape=(3,1)))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# model.summary()

# model.add(LSTM(50, return_sequences=True, input_shape=(3,1)))
# model.add(Dropout(0.2))  # 과적합을 피하기 위한 drop out 20% 설정
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mse', optimizer='rmsprop')

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, batch_size=1, epochs=100)

# predict 용 데이터
x_input = array([25,35,45])  # 1,3, ????
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)



