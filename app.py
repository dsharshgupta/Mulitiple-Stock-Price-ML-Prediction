import streamlit as st
import pandas as pd
import requests
from joblib import load,dump
import xgboost as xgb
import sklearn


def get_model(code):
    from datetime import datetime, timedelta
    end = str(datetime.now().year) + "-" + str(datetime.now().month).zfill(2) + "-" + str(datetime.now().day).zfill(2)
    da = datetime.now() - timedelta(days=365*20)
    start = str(da.year) + "-" + str(da.month).zfill(2) + "-" + str(da.day).zfill(2)
    url = f'https://api.upstox.com/v2/historical-candle/NSE_EQ%7C{code}/day/{end}/{start}'
    headers = {
        'Accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
    candels = data['data']['candles']
    candels_list = []
    for row in candels:
        date = row[0]
        open = row[1]
        high = row[2]
        low = row[3]
        close = row[4]
        volume = row[5]
        ele = {"date":date,"open":open,"high":high,"low":low,"close":close,"volume":volume}
        candels_list.append(ele)
    data_df = pd.DataFrame.from_dict(candels_list)
    data_df['date'] = pd.to_datetime(data_df['date']).dt.date
    data_df['previous_day_open'] = data_df['open'].shift(1)
    data_df['previous_day_low'] = data_df['low'].shift(1)
    data_df['previous_day_close'] = data_df['close'].shift(1)
    data_df['previous_day_volume'] = data_df['volume'].shift(1)
    data_df['next_day_high'] = data_df["high"]
    data_df = data_df.drop(columns=["open","low","high","close","date","volume"])
    data_df = data_df.dropna()
    X,y = data_df.drop(columns=['next_day_high']),data_df['next_day_high']
    model = xgb.XGBRegressor()
    model.fit(X,y)
    return model



def predict(var1, var2, var3, var4, model):
    l = ['previous_day_open','previous_day_low','previous_day_close','previous_day_volume']
    s = pd.DataFrame([[var1,var2,var3,var4]],columns=l,index=None)
    prediction = model.predict(s)[0]
    return prediction

def main():
    stock_options = {
        'Adani': 'INE002A01018',
        'IRCTC': 'INE335Y01020',
        'Indian Oil': 'INE242A01010',
        'Tata Power': 'INE245A01021',
        'Tata Motors': 'INE155A01022',
        'SBI':'INE062A01020'
    }

    selected_stock = st.selectbox('Select Stock', list(stock_options.keys()))

    var1 = st.number_input('Previous Day Open')
    var2 = st.number_input('Previous Day Low')
    var3 = st.number_input('Previous Day Close')
    var4 = st.number_input('Previous Day Volume')

    if st.button('Predict'):
        selected_code = stock_options[selected_stock]
        model = get_model(selected_code)
        with st.spinner('Predicting...'):
            prediction = predict(var1, var2, var3, var4, model)
        st.write(f'Predicted Price: {prediction}')

if __name__ == '__main__':
    main()