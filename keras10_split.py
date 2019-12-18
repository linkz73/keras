# sklearn 의 train_test_split 을 이용해 split 수행

#1. 데이터
import numpy as np
x = np.array(range(1, 101))  # (100,), ndarray
y = np.array(range(1, 101))
print(x)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# random_state : 랜덤시드를 부여해 랜덤 함수라 해도 검증 시에도 동일한 결과를 나오도록
# train_size, test_size 옵션을 둘다 적용시 train_size 이 적용됨. 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=3, test_size=0.5)

print(f"x_train:{len(x_train)}, x_test:{len(x_test)}, x_val:{len(x_val)}")

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, input_shape=(1, ), activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()

# 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
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