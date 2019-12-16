#### earlyStopping 적용하기 실습
# loss, acc, val_loss, val_acc
# keras05.py 를 카피해서 사용.

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, input_shape=(1, ), activation='relu'))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
# early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')
# model.fit(x_train, y_train, epochs=10000, callbacks=[early_stopping])

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='acc', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=10000, callbacks=[early_stopping])

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
# model.fit(x_train, y_train, epochs=10000, validation_data=(x_test, y_test), callbacks=[early_stopping])

# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# early_stopping = EarlyStopping(monitor='val_acc', patience=50, mode='auto')
# model.fit(x_train, y_train, epochs=10000, validation_data=(x_test, y_test), callbacks=[early_stopping])

loss = model.evaluate(x_test, y_test, batch_size=1)
# print(f"acc : ", acc)
print(f"loss : {loss}")

y_predict = model.predict(x_predict)
print(y_predict)