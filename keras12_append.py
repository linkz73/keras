#1. 데이터
import numpy as np
x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501, 601), range(711,811), range(100)])
x2 = np.array([range(100, 200), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
print(x.shape)
print(y.shape)

# x1 = np.transpose(x1)
x = x.T
y = y.T
print(x.shape)
print(y.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, random_state=3, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, shuffle=False, random_state=3, test_size=0.5)

print(f"x_train:{x_train.shape}, x_test:{len(x_test)}, x_val:{len(x_val)}")

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# 순차형 모델과 시퀀셜 모델의 차이 : 인풋과 아웃풋을 명시적으로 표현
input1 = Input(shape=(6,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(3)(dense3)
output1 = Dense(6)(dense4)

model = Model(inputs = input1, outputs = output1)
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)  #a[0], a[1]
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