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
TRADE_SYMBOL = 'BTCUSDT'
TIMEFRAME = Client.KLINE_INTERVAL_1MINUTE
SYTF = "https://api.binance.com/api/v1/klines?&symbol=BTCUSDT&interval=1m&limit=1550"
TRADE_QUANTITY = float(0.000603)

in_position1 = False
in_position2 = False

in_uptrend1 = True
in_uptrend2 = True

pricebuy = 0
pricebuy1 = 0

pricesell = 0
pricesell1 = 0


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
    return newestcandlesupertrend


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

    return donchian.DCL, donchian.DCM, donchian.DCU


##############################################################################################

# Main program
while True:

    # ping client to avoid timeout
    time.sleep(0.1)
    try:
        client = Client(api_key, api_secret)
        client.ping()
    except Exception:
        print("FALLA EL PING!")

    # Get Binance Data into dataframe
    try:
        candles = client.get_klines(symbol=TRADE_SYMBOL, interval=TIMEFRAME)
        df = pd.DataFrame(candles)
        df.columns = ['timestart', 'open', 'high', 'low', 'close', '?', 'timeend', '?', '?', '?', '?', 'ST']

        # Compute RSI after fixing data
        float_data = [float(x) for x in df.close.values]
        np_float_data = np.array(float_data)
        rsi = talib.RSI(np_float_data, 10)
        df['rsi'] = rsi

        upper, middle, lower = talib.BBANDS(np_float_data, 30)
        df['upper'] = upper
        df['middle'] = middle
        df['lower'] = lower

        sar = talib.SAR(df.high, df.low, acceleration=0.2, maximum=0.2)
        df['sar'] = sar

        macd, macdsignal, macdhist = talib.MACD(np_float_data, fastperiod=4, slowperiod=8, signalperiod=50)
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist

        ma50 = talib.MA(np_float_data, timeperiod=50)
        ma200 = talib.MA(np_float_data, timeperiod=200)
        df['ma50'] = ma50
        df['ma200'] = ma200

        # Compute StochRSI using RSI values in Stochastic function
        mystochrsi = Stoch(df.rsi, df.rsi, df.rsi, 3, 3, 10)
        df['MyStochrsiK'], df['MyStochrsiD'] = mystochrsi

        Aroondown, Aroonup = talib.AROON(df.high, df.low, 10)
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
        ST = df.ta.supertrend(length=20, multiplier=2)
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
        for current in range(1, len(df.index)):
            previous = current - 1

        last_row_index1 = len(df.index) - 1
        previous_row_index1 = last_row_index1 - 1

        last_row_index2 = len(df.index) - 2
        previous_row_index2 = last_row_index2 - 1

        last_row_index3 = len(df.index) - 3
        previous_row_index3 = last_row_index3 - 1

        last_row_index4 = len(df.index) - 4
        previous_row_index4 = last_row_index4 - 1

        last_row_index5 = len(df.index) - 5
        previous_row_index5 = last_row_index5 - 1


        def dchannel(df, window):

            DCU = df["high"].rolling(window=window).max()
            DCL = df["low"].rolling(window=window).min()

            return DataFrame({"DCU": DCU, "DCL": DCL})


        dchannel10 = dchannel(df, 7)
        df = pd.concat([df, dchannel10], axis=1)

        """
        print(df['MyStochrsiK'][previous_row_index1])
        print(df['MyStochrsiD'][previous_row_index1])

        print(df['close'][previous_row_index1])
        """

        time.sleep(0.1)
        # sec1

        # BUY*****************************
        if float(df['ST'][last_row_index1]) < float(df['close'][last_row_index1]):
            if float(df['ST'][previous_row_index1]) > float(df['close'][previous_row_index1]):
                # if float(df['MyStochrsiK'][last_row_index1]) < float(df['MyStochrsiD'][last_row_index1]):
                if float(df['arup'][last_row_index1]) > float(df['ardown'][last_row_index1]):
                    if float(df['macd'][last_row_index1]) > float(df['macdsignal'][last_row_index1]):
                        if float(df['macd'][previous_row_index1]) < float(df['macdsignal'][previous_row_index1]):
                            if float(df['MDM1'][last_row_index1]) < float(df['PDM1'][last_row_index1]):
                                if not in_position1:
                                    if not in_position2:
                                            print(
                                                "OVERLOW Process BUY BUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUYBUY")
                                            order = client.order_market_buy(
                                                symbol=TRADE_SYMBOL,
                                                quantity=TRADE_QUANTITY, )
                                            in_position1 = True
                                            pricebuy = float(newestcandleclose) - 95
                                            pricebuy1 = float(newestcandleclose) + 105

        time.sleep(0.1)

        # SELL*UPPER*****************************
        if float(df['high'][last_row_index1]) > float(pricebuy1):
            if in_position1:
                print(
                    "OVERHIGH Process SELL UPPER")
                balance = client.get_asset_balance(asset='BTC')
                formatted = '{0:.8}'.format(balance['free'])
                order = client.order_market_sell(
                    symbol=TRADE_SYMBOL,
                    quantity=formatted)
                in_position1 = False

        time.sleep(0.1)

        # SELL*DELAY1*****************************
        if float(df['sar'][last_row_index1]) > float(df['close'][last_row_index1]):
            if float(df['low'][last_row_index1]) < float(pricebuy):
                if in_position1:
                    print(
                        "OVERHIGH Process SELL DELAY1")
                    balance = client.get_asset_balance(asset='BTC')
                    formatted = '{0:.8}'.format(balance['free'])
                    order = client.order_market_sell(
                        symbol=TRADE_SYMBOL,
                        quantity=formatted)
                    in_position1 = False

        time.sleep(0.1)

    except Exception as e:
        print(e)
###############################################################





