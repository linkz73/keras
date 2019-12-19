# LSTM / RNN
# LSTM : 연속적인 데이터

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])  # (4,3)
'''
y 데이터 셋트가 별도로 주어지지 않고, x와 y가 같은 시계열 데이터에서 추출된 경우
x 의 마지막 세트는 삭제
y를 구성하는 방법 : 아래 공식대로 갯수가 정해짐. 
    1. 행의 갯수
        6개를 3개씩 자르는 경우 : 6-3 행 갯수
        6개를 4개씩 자르는 경우 : 6-4 행 갯수
    2. 잘라서 묶인 갯수를 y 데이터 앞에서 삭제한 나머지 전체 행 사용
''' 
y = array([4,5,6,7])  # (4,)
print(x)
print(y)
print("x.shape", x.shape)  # (4,3)
print("y.shape", y.shape)  # (4,3)

'''
 x   y
123  4
234  5
345  6
456  7
'''

# reshape 해서 전체를 곱한 값은 동일해야 함.
# 4,3 --> 4,3,1
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x)
print("x.shape", x.shape)  # (4,3,1)
'''
    x      y
[1][2][3]  4
[2][3][4]  5
[3][4][5]  6
[4][5][6]  7
'''

# 2. 모델구성
model = Sequential()
# input_shape(컬럼수, 컬럼을 몇개씩 잘라서 작업할 것인지)
model.add(LSTM(50, activation='relu', input_shape=(3,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1))
# model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, batch_size=1, epochs=100)

x_input = array([6,7,8])  # 1,3, ????
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)





