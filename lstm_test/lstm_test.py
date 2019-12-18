import pandas as pd
from pandas import DataFrame
import pandas.io.sql as pdsql
# from pandasql import sqldf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

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
    p = p.set_index(p['date'].values)
    p['date'] = pd.to_datetime(p['date'], errors='coerce').apply(lambda x:x.strftime('%Y%m%d'))
    pivots = peak_valley_pivots(p['close'].values, 0.02, -0.02)
    fig = plt.figure(constrained_layout=False,figsize=(20,7))
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    
    ts_pivots = pd.Series(p['close'], index=p.index)
    ts_pivots = ts_pivots[pivots != 0]

    p['date'] = p.index.map(mdates.date2num)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.plot(p.index, p['close'])
    ts_pivots.plot(style='g-o')
    plt.show()
    return pivots

'''
시작일자 = '2019-10-10'
종료일자 = '2019-11-30'
거래소 = "upbit"
종목 = 'BTC/KRW'
frame = '30m'
테이블 = f"data_{frame}_{거래소}"
'''

# df = get_price(종목, 시작일자, 종료일자, 테이블)
df = pd.read_csv('price/dataset/btcusdt_30m.csv')
print(df.head())

df['ZigZag'] = convZigzag(df)
# df['ZigZag'] = peak_valley_pivots(df['close'].values, 0.03, -0.03)
df['TREND'] = df['ZigZag'].replace(to_replace=0, method='ffill')
trends = df['TREND'].values
print(trends)

high_prices = df['high'].values
low_prices = df['low'].values
mid_prices = (high_prices + low_prices) / 2
seq_len = 50
sequence_length = seq_len + 1

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MaxAbsScaler()
# scaler.fit(x)  # fit 을 하면서 가중치가 생성됨.
# x = scaler.transform(x)

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])

def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

result = normalize_windows(result)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(Dropout(0.5))  # 과적합을 피하기 위한 drop out 20% 설정
# return_sequences=False 디폴트 값
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')
model.summary()


start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=20,
    callbacks=[
        # TensorBoard(log_dir='logs/%s' % (start_time)),
        ModelCheckpoint('./price/models/%s_eth.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
])

# 예측 결과를 그래프로 표현
pred = model.predict(x_test)
print(pred)
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()