# coding: utf-8
import pandas as pd
import talib
import numpy as np  # computing multidimensionla arrays
import datetime
import time
import settings
import pandas_ta as ta
import cfscrape
import json
from pandas import DataFrame

##############################################################################################
from binance.client import Client
from binance.enums import *

api_key = settings.api_key
api_secret = settings.api_secret
client = Client(api_key, api_secret)

STOCHRSI_OVERHIGH = 70
STOCHRSI_OVERLOW = 30
TRADE_SYMBOL = 'ILVUSDT'
TIMEFRAME = Client.KLINE_INTERVAL_1MINUTE
SYTF = "https://api.binance.com/api/v1/klines?&symbol=BTCUSDT&interval=1m&limit=1550"
TRADE_QUANTITY = 0.026

in_position1 = False
in_position2 = False

in_uptrend1 = True
in_uptrend2 = True


n = 0

pricebuy = 0
pricebuy1 = 0
pricebuy2 = 0

pricesell = 0
pricesell1 = 0
pricesell2 = 0

Profit = 0
Win = 0
Lose = 0


# Initialize Client and connect to Binance


# StochasticRSI Function
def Stoch(close, high, low, smoothk, smoothd, n):
    lowestlow = pd.Series.rolling(low, window=n, center=False).min()
    highesthigh = pd.Series.rolling(high, window=n, center=False).max()
    K = pd.Series.rolling(100 * ((close - lowestlow) / (highesthigh - lowestlow)), window=smoothk).mean()
    D = pd.Series.rolling(K, window=smoothd).mean()
    return K, D


def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True

def supertrend():
    BU = cfscrape.create_scraper()

    URL = SYTF

    ResultRaw = BU.get(URL, timeout=(10, 15)).content
    Result = json.loads(ResultRaw)

    for x in Result:
        TimeUnix = float(x[0]) / float(1000)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df = pd.DataFrame([x[:7] for x in Result],
                      columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ST'])

    format = '%Y-%m-%d %H:%M:%S'
    df['Date'] = pd.to_datetime(df['Date'], format=format)
    df["Open"] = pd.to_numeric(df["Open"], errors='coerce')
    df["High"] = pd.to_numeric(df["High"], errors='coerce')
    df["Low"] = pd.to_numeric(df["Low"], errors='coerce')
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    df = df.fillna(value=0)

    df['ST'] = df.ta.supertrend(length=20, multiplier=7)

    newestcandlesupertrend = df['ST'].astype(str).iloc[-1]
    return  newestcandlesupertrend

def donchian():
    BU = cfscrape.create_scraper()

    URL = SYTF

    ResultRaw = BU.get(URL, timeout=(10, 15)).content
    Result = json.loads(ResultRaw)

    for x in Result:
        TimeUnix = float(x[0]) / float(1000)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df = pd.DataFrame([x[:6] for x in Result],
                      columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    format = '%Y-%m-%d %H:%M:%S'
    df['Date'] = pd.to_datetime(df['Date'], format=format)
    df["Open"] = pd.to_numeric(df["Open"], errors='coerce')
    df["High"] = pd.to_numeric(df["High"], errors='coerce')
    df["Low"] = pd.to_numeric(df["Low"], errors='coerce')
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    df = df.fillna(value=0)

    df.ta.donchian(lower_length=7, upper_length=7, append=True)

    donchian.DCL = df['DCL_7_7'].iloc[-1]
    donchian.DCM = df['DCM_7_7'].iloc[-1]
    donchian.DCU = df['DCU_7_7'].iloc[-1]

    return  donchian.DCL, donchian.DCM, donchian.DCU
##############################################################################################

# Main program
while True:

    # ping client to avoid timeout
    client = Client(api_key, api_secret)

    # Get Binance Data into dataframe
    candles = client.get_historical_klines("ILVUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 Nov, 2021", "22 Nov, 2021")

    df = pd.DataFrame(candles)
    df.columns = ['timestart', 'open', 'high', 'low', 'close', '?', 'timeend', '?', '?', '?', '?', 'ST']

    # Compute RSI after fixing data
    float_data = [float(x) for x in df.close.values]
    np_float_data = np.array(float_data)
    rsi = talib.RSI(np_float_data, 5)
    df['rsi'] = rsi

    upper, middle, lower = talib.BBANDS(np_float_data, 30)
    df['upper'] = upper
    df['middle'] = middle
    df['lower'] = lower

    sar = talib.SAR(df.high, df.low, acceleration=0.05, maximum=0.2)
    df['sar'] = sar

    macd, macdsignal, macdhist = talib.MACD(np_float_data, fastperiod=5, slowperiod=10, signalperiod=10)
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist

    ema50 = talib.EMA(np_float_data, timeperiod=50)
    ema150 = talib.EMA(np_float_data, timeperiod=200)
    df['ema50'] = ema50
    df['ema150'] = ema150

    # Compute StochRSI using RSI values in Stochastic function
    mystochrsi = talib.STOCH(df.high, df.low, df.close, 5, 5, 5)
    df['MyStochrsiK'], df['MyStochrsiD'] = mystochrsi

    rsi1 = talib.RSI(np_float_data, 50)
    df['rsi1'] = rsi1

    mystochrsi1 = Stoch(df.rsi, df.rsi, df.rsi, 5, 5, 5)
    df['MyStochrsiK1'], df['MyStochrsiD1'] = mystochrsi1

    Aroondown, Aroonup = talib.AROON(df.high, df.low, 2)
    df['arup'] = Aroonup
    df['ardown'] = Aroondown

    MDM1 = talib.MINUS_DM(df.high, df.low, 2)
    df['MDM1'] = MDM1
    PDM1 = talib.PLUS_DM(df.high, df.low, 1)
    df['PDM1'] = PDM1

    MDM2 = talib.MINUS_DM(df.high, df.low, 1)
    df['MDM2'] = MDM2
    PDM2 = talib.PLUS_DM(df.high, df.low, 2)
    df['PDM2'] = PDM2

    ADX = talib.ADX(df.high, df.low, df.close, 10)
    df['ADX'] = ADX

    df = df.astype(float)
    ST = df.ta.supertrend(length=20, multiplier=5)
    df['ST'] = ST

    AO = df.ta.ao(4, 8)
    df['AO'] = AO


    df.ta.fisher(9, 9, append=True)

    df['FI'] = df['FISHERT_9_9']
    df['FL'] = df['FISHERTs_9_9']
    #################################### End of Main #############################################
    # WARNING: If Logging is removed uncomment the next line.
    # time.sleep(1) # Sleep for 1 second. So IP is not rate limited. Can be faster. Up to 1200 requests per minute.

    #################################### Logging #################################################
    newestcandlestart = df.timestart.astype(str).iloc[-1]  # gets last time
    newestcandleclose = df.close.iloc[-1]  # gets last close
    newestcandleRSI = df.rsi.astype(str).iloc[-1]  # gets last rsi
    newestcandleK = df.MyStochrsiK.astype(str).iloc[-1]
    newestcandleD = df.MyStochrsiD.astype(str).iloc[-1]
    newestcandleupper = df.upper.astype(str).iloc[-1]
    newestcandlemiddle = df.middle.astype(str).iloc[-1]
    newestcandlelower = df.lower.astype(str).iloc[-1]
    newestcandlesar = df.sar.astype(str).iloc[-1]
    newestcandlemacd = df.macd.astype(str).iloc[-1]
    newestcandlemacdsignal = df.macdsignal.astype(str).iloc[-1]
    newestcandlemacdhist = df.macdhist.astype(str).iloc[-1]


    """
    print("Date" + '         ' + newestcandlestart)
    print("Price" + '        ' + newestcandleclose)
    print("RSI" + '          ' + newestcandleRSI)
    print("%K" + '           ' + newestcandleK)
    print("%D" + '           ' + newestcandleD)
    print("Upper" + '        ' + newestcandleupper)
    print("Middle" + '       ' + newestcandlemiddle)
    print("Lower" + '        ' + newestcandlelower)
    print("Sar" + '          ' + newestcandlesar)
    print("MACD" + '         ' + newestcandlemacd)
    print("MACDSignal" + '   ' + newestcandlemacdsignal)
    print("MACDHist" + '     ' + newestcandlemacdhist)
    print("MA9" + '         ' + newestcandlema9)
    print("MA50" + '        ' + newestcandlema50)

    print('')
    print(df.tail(5))
    print(df.tail(100))
    """

    def dchannel(df, window):
         DCU = df["high"].rolling(window=window).max()
         DCL = df["low"].rolling(window=window).min()

         return DataFrame({"DCU": DCU, "DCL": DCL})


    dchannel10 = dchannel(df, 7)
    df = pd.concat([df, dchannel10], axis=1)


    for current in range(1, len(df.index)):
        previous = 10000

        n = n + 1
        previous1 = current + previous - n

        last_row_index1 = len(df.index) - previous1 + n
        previous_row_index1 = len(df.index) - previous1 + n -1
        previous_row_index2 = len(df.index) - previous1 + n -2
        previous_row_index3 = len(df.index) - previous1 + n -3
        previous_row_index4 = len(df.index) - previous1 + n -4
        previous_row_index5 = len(df.index) - previous1 + n -5

        last24_row_index1 = last_row_index1 - 400
        priceChangePercent = round(float(df['close'][last_row_index1]) * 100 / float(df['close'][last24_row_index1]) - 100, 2)
        print(priceChangePercent)

        # sec1

        # BUY*****************************
        if float(df['ema150'][previous_row_index1]) < float(df['close'][previous_row_index1]):
                #if float(df['MyStochrsiK'][last_row_index1]) and float(df['MyStochrsiD'][last_row_index1]) < STOCHRSI_OVERLOW:
                    if float(df['low'][previous_row_index1]) < float(df['lower'][previous_row_index1]):
                        #if float(df['MyStochrsiK'][previous_row_index1]) < float(df['MyStochrsiD'][previous_row_index1]):
                            #if float(df['ema50'][previous_row_index1]) > float(df['ema150'][previous_row_index1]):
                                        if not in_position1:
                                            if not in_position2:
                                                            print("OVERLOW BUY")
                                                            usdt = 20
                                                            price = (float(df['open'][last_row_index1]) + float(df['high'][last_row_index1])) / 2
                                                            amount = 40 / price
                                                            amountfee = amount * 0.999
                                                            in_position1 = True
                                                            in_uptrend1 = False
                                                            pricebuy = price - 20
                                                            pricebuy1 = price + 10

        # SELL*UPPER*****************************
        if float(df['high'][last_row_index1]) > float(pricebuy1):
            if not in_uptrend1:
                if in_position1:
                    print(
                        "OVERHIGH Process SELL UPPER")

                    Profit = Profit + 0.11 - 0.03
                    Win = Win + 1
                    in_position1 = False
                    in_uptrend1 = True
                else:
                    print("It is overhigh, UPPER")

        # SELL*DELAY1*****************************
        if float(df['ema150'][previous_row_index1]) > float(df['close'][previous_row_index1]):
                if not in_uptrend1:
                    if in_position1:
                        print(
                            "OVERHIGH Process SELL DELAY1")

                        Profit = Profit - 0.11 - 0.03
                        Lose = Lose + 1
                        in_position1 = False
                        in_uptrend1 = True
                    else:
                        print("It is overhigh, DELAY1")

        print('Profit' + '   ' + str(Profit))
        print('Win' + '      ' + str(Win))
        print('Lose' + '     ' + str(Lose))
        """
        # SELL*DELAY2*****************************
        if float(df['sar'][last_row_index1]) > float(df['close'][last_row_index1]):
            if float(df['ST'][last_row_index1]) > float(df['close'][last_row_index1]):
                if not in_uptrend1:
                    if in_position1:
                        print(
                        "OVERHIGH Process SELL DELAY2")
                        pricesell = pricebuy2 -
                        in_position1 = False
                        in_uptrend1 = True
                        time.sleep(60)
                    else:
                        print("It is overhigh, DELAY2")

        time.sleep(0.2)
        """


###############################################################





