import gc
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def get_market_data(market, tag=True):
    """
    market: the full name of the cryptocurrency as spelled on coinmarketcap.com. eg.: 'bitcoin'
    tag: eg.: 'btc', if provided it will add a tag to the name of every column.
    returns: panda DataFrame
    This function will use the coinmarketcap.com url for provided coin/token page. 
    Reads the OHLCV and Market Cap.
    Converts the date format to be readable. 
    Makes sure that the data is consistant by converting non_numeric values to a number very close to 0.
    And finally tags each columns if provided.
    """
    # market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market + 
    #                             "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]
    market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market + 
                                "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"), flavor='html5lib')[2]
    market_data = market_data.assign(Date=pd.to_datetime(market_data['Date']))  
    market_data['Volume'] = (pd.to_numeric(market_data['Volume'], errors='coerce').fillna(0))
    if tag:
        market_data.columns = [market_data.columns[0]] + [tag + '_' + i for i in market_data.columns[1:]]
    return market_data

btc_data = get_market_data("bitcoin", tag='BTC')
print(btc_data.head())
show_plot(btc_data, tag='BTC')
# https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=


def merge_data(a, b, from_date=merge_date):
    """
    a: first DataFrame
    b: second DataFrame
    from_date: includes the data from the provided date and drops the any data before that date.
    returns merged data as Pandas DataFrame
    """
    merged_data = pd.merge(a, b, on=['Date'])
    merged_data = merged_data[merged_data['Date'] >= from_date]
    return merged_data


def add_volatility(data, coins=['BTC', 'ETH']):
    """
    data: input data, pandas DataFrame
    coins: default is for 'btc and 'eth'. It could be changed as needed
    This function calculates the volatility and close_off_high of each given coin in 24 hours, 
    and adds the result as new columns to the DataFrame.
    Return: DataFrame with added columns
    """
    for coin in coins:
        # calculate the daily change
        kwargs = {coin + '_change': lambda x: (x[coin + '_Close'] - x[coin + '_Open']) / x[coin + '_Open'],
                coin + '_close_off_high': lambda x: 2*(x[coin + '_High'] - x[coin + '_Close']) / (x[coin + '_High'] - x[coin + '_Low']) - 1,
                coin + '_volatility': lambda x: (x[coin + '_High'] - x[coin + '_Low']) / (x[coin + '_Open'])}
        data = data.assign(**kwargs)
    return data


def create_model_data(data):
    """
    data: pandas DataFrame
    This function drops unnecessary columns and reverses the order of DataFrame based on decending dates.
    Return: pandas DataFrame
    """
    #data = data[['Date']+[coin+metric for coin in ['btc_', 'eth_'] for metric in ['Close','Volume','close_off_high','volatility']]]
    data = data[['Date']+[coin+metric for coin in ['BTC_', 'ETH_'] for metric in ['Close','Volume']]]
    data = data.sort_values(by='Date')
    return data


def split_data(data, training_size=0.8):
    """
    data: Pandas Dataframe
    training_size: proportion of the data to be used for training
    This function splits the data into training_set and test_set based on the given training_size
    Return: train_set and test_set as pandas DataFrame
    """
    return data[:int(training_size*len(data))], data[int(training_size*len(data)):]


def create_inputs(data, coins=['BTC', 'ETH'], window_len=window_len):
    """
    data: pandas DataFrame, this could be either training_set or test_set
    coins: coin datas which will be used as the input. Default is 'btc', 'eth'
    window_len: is an intiger to be used as the look back window for creating a single input sample.
    This function will create input array X from the given dataset and will normalize 'Close' and 'Volume' between 0 and 1
    Return: X, the input for our model as a python list which later needs to be converted to numpy array.
    """
    norm_cols = [coin + metric for coin in coins for metric in ['_Close', '_Volume']]
    inputs = []
    for i in range(len(data) - window_len):
        temp_set = data[i:(i + window_len)].copy()
        inputs.append(temp_set)
        for col in norm_cols:
            inputs[i].loc[:, col] = inputs[i].loc[:, col] / inputs[i].loc[:, col].iloc[0] - 1  
    return inputs


def create_outputs(data, coin, window_len=window_len):
    """
    data: pandas DataFrame, this could be either training_set or test_set
    coin: the target coin in which we need to create the output labels for
    window_len: is an intiger to be used as the look back window for creating a single input sample.
    This function will create the labels array for our training and validation and normalize it between 0 and 1
    Return: Normalized numpy array for 'Close' prices of the given coin
    """
    return (data[coin + '_Close'][window_len:].values / data[coin + '_Close'][:-window_len].values) - 1


def to_array(data):
    """
    data: DataFrame
    This function will convert list of inputs to a numpy array
    Return: numpy array
    """
    x = [np.array(data[i]) for i in range (len(data))]
    return np.array(x)


train_set = train_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)
X_train = create_inputs(train_set)
Y_train_btc = create_outputs(train_set, coin='BTC')
X_test = create_inputs(test_set)
Y_test_btc = create_outputs(test_set, coin='BTC')
Y_train_eth = create_outputs(train_set, coin='ETH')
Y_test_eth = create_outputs(test_set, coin='ETH')
X_train, X_test = to_array(X_train), to_array(X_test)

def build_model(inputs, output_size, neurons, activ_func=activation_function, dropout=dropout, loss=loss, optimizer=optimizer):
    """
    inputs: input data as numpy array
    output_size: number of predictions per input sample
    neurons: number of neurons/ units in the LSTM layer
    active_func: Activation function to be used in LSTM layers and Dense layer
    dropout: dropout ration, default is 0.25
    loss: loss function for calculating the gradient
    optimizer: type of optimizer to backpropagate the gradient
    This function will build 3 layered RNN model with LSTM cells with dripouts after each LSTM layer 
    and finally a dense layer to produce the output using keras' sequential model.
    Return: Keras sequential model and model summary
    """
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model

neurons = 512                 
activation_function = 'tanh'  
loss = 'mse'                  
optimizer="adam"              
dropout = 0.25                 
batch_size = 12               
epochs = 53                   
window_len = 7               
training_size = 0.8
merge_date = '2016-01-01'

# clean up the memory
gc.collect()
# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
btc_model = build_model(X_train, output_size=1, neurons=neurons)
# train model on data
btc_history = btc_model.fit(X_train, Y_train_btc, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test_btc), shuffle=False)