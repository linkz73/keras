from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([11,12,13,14,15])

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

'''
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=100)

# mean squd error, batch-size:1개 단위, 작을 수록 정확도 상승
loss, acc = model.evaluate(x, y)
print(f"acc : ", acc)
print(f"loss : {loss:0.14f}")

y_predict = model.predict(x2)
print(y_predict)
'''