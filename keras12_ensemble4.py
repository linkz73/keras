# input 1 -> output 2

# 1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])

y1 = np.array([range(501, 601), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

# x1 = np.transpose(x1)
x1 = x1.T
y1 = y1.T
y2 = y2.T

print(x1.shape)
print(y1.shape)
print(y2.shape)

# train : test : val = 6 : 2 : 2
from sklearn.model_selection import train_test_split
x1_train, x1_test = train_test_split(x1, shuffle=False, random_state=3, test_size=0.4)
x1_val, x1_test = train_test_split(x1_test, shuffle=False, random_state=3, test_size=0.5)
y1_train, y1_test, y2_train, y2_test = train_test_split(y1, y2, shuffle=False, random_state=3, test_size=0.4)
y1_val, y1_test, y2_val, y2_test = train_test_split(y1_test, y2_test, random_state=3, test_size=0.5)

print(f"x1_train:{x1_train.shape}, x1_test:{x1_test.shape}, x1_val:{x1_val.shape}")
print(f"y1_train:{y1_train.shape}, y1_test:{y1_test.shape}, y1_val:{y1_val.shape}")
print(f"y2_train:{y2_train.shape}, y2_test:{y2_test.shape}, y2_val:{y2_val.shape}")

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# 순차형 모델과 시퀀셜 모델의 차이 : 인풋과 아웃풋을 명시적으로 표현
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(10)(dense1)
dense3 = Dense(4)(dense2)
middle1 = Dense(3)(dense3)

output1 = Dense(5)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(5)(output1)
output1 = Dense(3)(output1)

output2 = Dense(5)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(5)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = input1, outputs = [output1, output2])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train, [y1_train,y2_train], epochs=100, batch_size=1, validation_data=(x1_val, [y1_val,y2_val]))

# 4. 평가 예측
mse = model.evaluate(x1_test, [y1_test,y2_test], batch_size=1)  #a[0], a[1]
print("mse1 : ", mse)

# 모델의 갯수 만큼 mse가 리스트 형태로 출력됨.
# print("loss1 : ", loss)

y1_predict,y2_predict = model.predict(x1_test)
print(y1_predict)
print(y2_predict)

# RMSE 구하는 수식 추가
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE-1 : ", RMSE1)
print("RMSE-2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print("R2-1 : ", r2_y1_predict)
print("R2-2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict)/2)