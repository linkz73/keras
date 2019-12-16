import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터

b = np.array(range(1,11))

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
        # temp = []
        # for item in subset:
        #     temp.append(item)
        # aaa.append(temp)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(b, size)
print("================")
print(dataset)

x_train = dataset[:,0:-1]
y_train = dataset[:,-1]
# print(x_train)
print(x_train.shape)  # (6,4)
# print(y_train)
print(y_train.shape)  # (6,)

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# train, test 둘다 적용시 train 이 적용됨.

model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

# 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가 예측
loss, mse = model.evaluate(x_train, y_train, batch_size=1, verbose=2)  #a[0], a[1]
print("mse : ", mse)
print("loss : ", loss)

x2_array = np.array([7,8,9,10])  # (4,) --> (1,4)
print("x2_array", x2_array.shape)
x2_array = x2_array.reshape((1,4))
print("x2_array", x2_array.shape)
y_predict = model.predict(x2_array)
print(y_predict)
