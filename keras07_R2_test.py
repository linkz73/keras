# R2 : regression 모델의 정확도를 나타내며, 1이 최고값.
# RMSE : regression 모델의 경우 오류를 측정하는 지표, 낮을 수록 우수, 0이 최고값

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  #(10,) 벡터
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(1000, input_shape=(1, ), activation='relu'))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# acc :  1.0 , loss :  2.7284841053187847e-12
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=200, batch_size=1)

loss, mse = model.evaluate(x_test, y_test)  #a[0], a[1]
print("mse : ", mse)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하는 수식 추가
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# 문제1. R2를 0.5이하로 줄이시오.
# 레이어는 인풋과 아웃풋 포함 5개 이상, 노드는 각 레이어당 5개 이상
# batch_size = 1
# epochs = 100 이상


# 실행결과
"""
mse :  3.3843331336975098
loss :  3.3843331336975098
[[ 9.782363]
 [10.639115]
 [11.499335]
 [12.362111]
 [13.230191]
 [14.099338]
 [14.98854 ]
 [15.884316]
 [16.781414]
 [17.679428]]
RMSE :  1.8396556624803262
R2 :  0.589777823455039
"""