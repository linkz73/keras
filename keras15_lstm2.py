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
print(x_train.shape)  # (6,4)
print(y_train.shape)  # (6,)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
print(x_train)
print(x_train.shape)

# 2. 모델
model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 예측
loss, mse = model.evaluate(x_train, y_train, batch_size=1, verbose=2)  #a[0], a[1]
print("mse : ", mse)

x2_array = np.array([7,8,9,10])  # (4,) --> (1,4,1)
print("x2_array", x2_array.shape)
x2_array = x2_array.reshape((1,4,1))
print("x2_array", x2_array.shape)
y_predict = model.predict(x2_array)
print(y_predict)
