#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501, 601), range(711,811), range(100)])

x2 = np.array([range(100, 200), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

# x1 = np.transpose(x1)
x1 = x1.T
y1 = y1.T
x2 = x2.T
y2 = y2.T
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=3, test_size=0.4)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=3, test_size=0.5)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=3, test_size=0.4)
x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=3, test_size=0.5)

print(f"x1_train:{x1_train.shape}, x1_test:{len(x1_test)}, x1_val:{len(x1_val)}")
print(f"x2_train:{x2_train.shape}, x2_test:{len(x2_test)}, x2_val:{len(x2_val)}")

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# 순차형 모델과 시퀀셜 모델의 차이 : 인풋과 아웃풋을 명시적으로 표현
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(3)(dense3)
middle1 = Dense(3)(dense4)

input2 = Input(shape=(3,))  #  shape는 컬럼 형태 표시
xx = Dense(5, activation='relu')(input1)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
middle2 = Dense(3)(xx)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])  # output 만 묶어 주면 됨.

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train, y1_train, epochs=100, batch_size=1, validation_data=(x1_val, y1_val))

# 4. 평가 예측
loss, mse = model.evaluate(x1_test, y1_test, batch_size=1)  #a[0], a[1]
print("mse : ", mse)
print("loss : ", loss)

y_predict = model.predict(x1_test)
print(y_predict)

# RMSE 구하는 수식 추가
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y_predict)
print("R2 : ", r2_y_predict)