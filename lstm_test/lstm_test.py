import pandas as pd
from pandas import DataFrame
import pandas.io.sql as pdsql
from pandasql import sqldf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Activation, Input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from zigzag import *
import configparser
import warnings
import sys
warnings.filterwarnings("ignore")

def comma_volume(x, pos):  # formatter function takes tick label and tick position
    s = '{:0,d}K'.format(int(x/1000))
    return s

def comma_price(x, pos):  # formatter function takes tick label and tick position
    s = '{:0,d}'.format(int(x))
    return s

def comma_percent(x, pos):  # formatter function takes tick label and tick position
    s = '{:+.2f}'.format(x)
    return s


# # zigzag 항목 추가
# def convZigzag(p):    
#     # from zigzag import *
#     p = p.set_index(p['date'].values)
#     p['date'] = pd.to_datetime(p['date'], errors='coerce').apply(lambda x:x.strftime('%Y%m%d'))
#     pivots = peak_valley_pivots(p['close'].values, 0.03, -0.03)
#     fig = plt.figure(constrained_layout=False,figsize=(20,7))
#     gs = GridSpec(1, 1, figure=fig)
#     ax1 = fig.add_subplot(gs[0])
    
#     ts_pivots = pd.Series(p['close'], index=p.index)
#     ts_pivots = ts_pivots[pivots != 0]
    
#     p['date'] = p.index.map(mdates.date2num)
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax1.plot(p.index, p['close'])
#     ts_pivots.plot(style='g-o')
#     return pivots

def convZigzag(p):    
    # p = p.set_index(p['date'].values)
    p = p.set_index(p['date'])
    print("p : ", p)
    p['date'] = pd.to_datetime(p['date'], errors='coerce').apply(lambda x:x.strftime('%Y%m%d'))
    pivots = peak_valley_pivots(p['close'].values, 0.02, -0.02)
    fig = plt.figure(constrained_layout=False,figsize=(20,7))
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    
    ts_pivots = pd.Series(p['close'], index=p.index)
    ts_pivots = ts_pivots[pivots != 0]

    # p['date'] = p.index.map(mdates.date2num)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax1.plot(p.index, p['close'])
    # ts_pivots.plot(style='g-o')
    # plt.show()
    return pivots

시작일자 = '2019-10-10'
종료일자 = '2019-11-30'

# df = get_price(종목, 시작일자, 종료일자, 테이블)
df = pd.read_csv('lstm_test/dataset/btcusdt_30m.csv')

query_tmp = f"SELECT * FROM df WHERE date > '{시작일자}' AND date < '{종료일자}'"
df = sqldf(query_tmp, locals())
# last_day = df_tmp.values[0][0][0:10]
print(df.head())

# df['ZigZag'] = convZigzag(df)
df['ZigZag'] = peak_valley_pivots(df['close'].values, 0.03, -0.03)
df['TREND'] = df['ZigZag'].replace(to_replace=0, method='ffill')

from keras.utils import np_utils

# 1. 데이터 전처리
X1 = df['close'].values
X2 = df['volume'].values
y = df['TREND'].values
print(y.shape)  # (60000,)

size = 50
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

X1 = split_x(X1, size)
X1 = X1[:-1,:]
X2 = split_x(X2, size)
X2 = X2[:-1,:]
# y 값은 x를 lstm 모델에 맞도록 앞에서 사이즈만큼 잘라주어야 함.
y = y[size:,]

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x)  # fit 을 하면서 가중치가 생성됨.
# x = scaler.transform(x)
X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)

X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))
X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], 1))
# y = np.reshape(y, (y.shape[0], y.shape[1], 1))
print(X1)
print(X2)
print(y)
print(y.shape)
# -1,1 값의 범위를 0,1 로 변환
y = (y + 1)/2
# high_prices = df['high'].values
# low_prices = df['low'].values
# mid_prices = (high_prices + low_prices) / 2
# seq_len = 50
# sequence_length = seq_len + 1

from sklearn.model_selection import train_test_split
X1_train, X1_test, X2_train, X2_test = train_test_split(X1, X2, random_state=3, test_size=0.4)
X1_val, X1_test, X2_val, X2_test = train_test_split(X1_test, X2_test, random_state=3, test_size=0.5)
y_train, y_test = train_test_split(y, random_state=3, test_size=0.4)
y_val, y_test = train_test_split(y_test, random_state=3, test_size=0.5)

y1_train = np_utils.to_categorical(y_train)
y1_test = np_utils.to_categorical(y_test)
y1_val = np_utils.to_categorical(y_val)

print("X1_train shape", X1_train.shape)  # (1062,50,1)
print("X2_train shape", X2_train.shape)  # (1062,50,1)
print("X1_test shape", X1_test.shape)  # (355,50,1)
print("X2_test shape", X2_test.shape)  # (354,50,1)
print("X1_val shape", X1_val.shape)  # (355,50,1)
print("X2_val shape", X2_val.shape)  # (354,50,1)
print("y_train shape", y_train.shape)  # (1062,2)
print("y_test shape", y1_test.shape)  # (355,2)
print("y_val shape", y_val.shape)  # (354,2)

input1 = Input(shape=(50,1))
layers = LSTM(200, return_sequences=True, activation='relu')(input1)
layers = LSTM(64, activation='relu')(layers)
middle1 = Dropout(0.5)(layers)

input2 = Input(shape=(50,1))
layers2 = LSTM(120, return_sequences=True, activation='relu')(input2)
layers2 = LSTM(64, activation='relu')(layers2)
middle2 = Dropout(0.5)(layers2)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])  # output 만 묶어 주면 됨.
# output1 = Dense(1, activation='relu')(merge1)
output1 = Dense(2, activation='softmax')(merge1)

model = Model(inputs = [input1, input2], outputs = output1)
model.summary()

from keras.callbacks import EarlyStopping

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=10)

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model.fit([X1_train,X2_train], y1_train,
    validation_data=([X1_val,X2_val], y1_val),
    batch_size=10,
    verbose=2,
    epochs=1000,
    callbacks=[
        # TensorBoard(log_dir='logs/%s' % (start_time)),
        early_stopping_callback,
        ModelCheckpoint('./lstm_test/models/%s_eth.h5' % (start_time), monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
])


# 예측 결과를 그래프로 표현
pred = model.predict([X1_test,X2_test])
pred = list(map(lambda x:np.argmax(x), pred))
# print("\n Test Accuracy: %.4f" % (model.evaluate([X1_test,X2_test], y_test)[0]))
print(pred)
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()