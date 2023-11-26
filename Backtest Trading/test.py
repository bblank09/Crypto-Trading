import ccxt
from ccxt import binance

import settings
import pandas as pd
import cfscrape
import pandas_ta as ta
import json

api_key = settings.api_key
api_secret = settings.api_secret

pd.set_option('display.max_rows', None)

import warnings

warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime
import time

TRADE_SYMBOL = 'BNBUP/USDT'
TRADE_QUANTITY = 0.04
TIMEFRAME = '1m'
in_position = False

exchange = ccxt.binance({
    "apiKey": settings.api_key,
    "secret": settings.api_secret
})


def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])

    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)

    return tr


def atr(data, period):
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()

    return atr


def supertrend(df, period=20, atr_multiplier=20):
    BU = cfscrape.create_scraper()

    URL = "https://api.binance.com/api/v1/klines?&symbol=BNBUPUSDT&interval=1m&limit=1550"

    ResultRaw = BU.get(URL, timeout=(10, 15)).content
    Result = json.loads(ResultRaw)

    for x in Result:
        TimeUnix = float(x[0]) / float(1000)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df = pd.DataFrame([x[:8] for x in Result],
                      columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ST', 'in_uptrend'])

    format = '%Y-%m-%d %H:%M:%S'
    df['Date'] = pd.to_datetime(df['Date'], format=format)
    df["Open"] = pd.to_numeric(df["Open"], errors='coerce')
    df["High"] = pd.to_numeric(df["High"], errors='coerce')
    df["Low"] = pd.to_numeric(df["Low"], errors='coerce')
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    df = df.fillna(value=0)

    df['ST'] = df.ta.supertrend(length=20, multiplier=20)

    for current in range(1, len(df.index)):
        previous = current - 1

        if df['close'][current] > df['ST'][previous]:
            df['in_uptrend'][current] = True
        elif df['close'][current] < df['ST'][previous]:
            df['in_uptrend'][current] = False
        else:
            df['in_uptrend'][current] = df['in_uptrend'][previous]

            if df['in_uptrend'][current] and df['ST'][current] < df['ST'][previous]:
                df['ST'][current] = df['ST'][previous]

            if not df['in_uptrend'][current] and df['ST'][current] > df['ST'][previous]:
                df['ST'][current] = df['ST'][previous]

    return df




def check_buy_sell_signals(df):
    global in_position

    print("checking for buy and sell signals")
    print(df.tail(5))
    last_row_index = len(df.index) - 2
    previous_row_index = last_row_index - 2

    # sec1###########

    if in_position == False:
        in_position = False
        
    if in_position == True:
        in_position = True
        
    if not df['in_uptrend'][previous_row_index] and df['in_uptrend'][last_row_index]:
        print("changed to uptrend, buy")
        if not in_position:
            order = exchange.create_market_buy_order(TRADE_SYMBOL, TRADE_QUANTITY)
            print(order)
            in_position = True
        else:
            print("already in position, nothing to do")

    if df['in_uptrend'][previous_row_index] and not df['in_uptrend'][last_row_index]:
        print("changed to downtrend, sell")
        if in_position:
            order = exchange.create_market_sell_order(TRADE_SYMBOL, TRADE_QUANTITY)
            print(order)
            in_position = False
        else:
            print("You aren't in position, nothing to sell")


def run_bot():
    print(f"Fetching new bars for {datetime.now().isoformat()}")
    bars = exchange.fetch_ohlcv(TRADE_SYMBOL, timeframe=TIMEFRAME, limit=100)
    df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    supertrend_data = supertrend(df)
    check_buy_sell_signals(supertrend_data)



while True:
    run_bot()
    time.sleep(1)
