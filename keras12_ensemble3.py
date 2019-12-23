# input 2 -> output 1

#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(100, 200), range(311,411), range(100,200)])

y = np.array([range(501, 601), range(311,411), range(100,200)])

# x1 = np.transpose(x1)
# transpose는 벡터 형태를 제외하고 적용 가능
x1 = x1.T
x2 = x2.T
y = y.T

print(x1.shape)  #(100,3)
print(x2.shape)  #(100,3)
print(y.shape)  #(100,3)

# train : test : val = 6 : 2 : 2
from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, shuffle=False, random_state=3, test_size=0.4)
# x1_val, x1_test, x2_val, x2_test = train_test_split(x1_test, x2_test, random_state=3, test_size=0.5)
# y_train, y_test = train_test_split(y, shuffle=False, random_state=3, test_size=0.4)
# y_val, y_test = train_test_split(y_test, shuffle=False, random_state=3, test_size=0.5)

# train_test_split 는 행 갯수가 동일해야 함
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, shuffle=False, random_state=3, test_size=0.4)
x1_val, x1_test, y_val, y_test = train_test_split(x1_test, y_test, random_state=3, test_size=0.5)
x2_train, x2_test = train_test_split(x2, shuffle=False, random_state=3, test_size=0.4)
x2_val, x2_test = train_test_split(x2_test, shuffle=False, random_state=3, test_size=0.5)

print(f"x1_train:{x1_train.shape}, x1_test:{x1_test.shape}, x1_val:{x1_val.shape}")
print(f"x2_train:{x2_train.shape}, x2_test:{x2_test.shape}, x2_val:{x2_val.shape}")
print(f"y_train:{y_train.shape}, y_test:{y_test.shape}, y_val:{y_val.shape}")

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# 순차형 모델과 시퀀셜 모델의 차이 : 인풋과 아웃풋을 명시적으로 표현
input1 = Input(shape=(3,))
dense = Dense(5, activation='relu')(input1)
dense = Dense(10)(dense)
dense = Dense(4)(dense)
middle1 = Dense(3)(dense)

input2 = Input(shape=(3,))  #  shape는 컬럼 형태 표시
dense = Dense(5, activation='relu')(input2)
dense = Dense(10)(dense)
dense = Dense(4)(dense)
middle2 = Dense(3)(dense)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])  # output 만 묶어 주면 됨.

output1 = Dense(5)(merge1)
output1 = Dense(10)(output1)
output1 = Dense(5)(output1)
output1 = Dense(3)(output1)

model = Model(inputs = [input1, input2], outputs = output1)
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train,x2_train], y_train, epochs=100, batch_size=1, validation_data=([x1_val,x2_val], y_val))

# 4. 평가 예측
mse = model.evaluate([x1_test,x2_test], y_test, batch_size=1)  #a[0], a[1]
print("mse1 : ", mse)

# 모델의 갯수 만큼 mse가 리스트 형태로 출력됨.
# print("loss1 : ", loss)

# 인풋2, 아웃풋1 앙상블 모델이므로 아웃풋은 1
y_predict = model.predict([x1_test,x2_test])
print(y_predict)

# RMSE 구하는 수식 추가
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)
print("RMSE : ", RMSE)

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)