import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Algoritmo di Trading RSI + XGBoost")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker", value="NVDA")
period = st.sidebar.selectbox("Periodo", ["1y", "2y", "3y", "5y"], index=2)
profit_threshold = st.sidebar.slider("Soglia profitto target (%)", 0.5, 5.0, 1.0) / 100

# Carica dati
@st.cache_data
def load_data(ticker, period):
    return yf.download(ticker, period=period, interval="1d")

data = load_data(ticker, period)

if data.empty:
    st.error(f"Nessun dato disponibile per il ticker '{ticker}' nel periodo '{period}'.")
    st.stop()

# Controllo colonna 'Close'
if 'Close' not in data.columns or data['Close'].isnull().all():
    st.error("Errore: il dataset non contiene dati validi nella colonna 'Close'.")
    st.write("Ecco le prime righe dei dati scaricati:", data.head())
    st.stop()

# Calcolo RSI con gestione NaN
try:
    rsi_indicator = RSIIndicator(close=data['Close'].fillna(method='ffill'), window=14)
    data['RSI'] = rsi_indicator.rsi()
except Exception as e:
    st.error(f"Errore nel calcolo dell'RSI: {e}")
    st.stop()

# Calcoli aggiuntivi
data['Return'] = data['Close'].pct_change().shift(-1)
data['Target'] = (data['Return'] > profit_threshold).astype(int)
data.dropna(inplace=True)

# Features
X = data[['RSI']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modello
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Segnali e backtest
signals = pd.DataFrame(index=data.index[-len(y_test):])
signals['Prediction'] = y_pred
signals['RSI'] = X_test['RSI']
signals['Price'] = data.loc[signals.index, 'Close']
signals['Buy_Signal'] = (signals['Prediction'] == 1) & (signals['RSI'] < 30)
signals['Sell_Signal'] = (signals['Prediction'] == 0) & (signals['RSI'] > 70)
signals['Position'] = 0
signals.loc[signals['Buy_Signal'], 'Position'] = 1
signals['Daily_Return'] = data.loc[signals.index, 'Close'].pct_change().fillna(0)
signals['Strategy_Return'] = signals['Position'].shift(1) * signals['Daily_Return']
signals['Cumulative_Strategy_Return'] = (1 + signals['Strategy_Return']).cumprod()

# ROI
initial_investment = 1000
final_value = initial_investment * signals['Cumulative_Strategy_Return'].iloc[-1]
roi_percentage = (final_value - initial_investment) / initial_investment * 100

st.metric("Valore finale", f"{final_value:.2f} €")
st.metric("ROI stimato", f"{roi_percentage:.2f} %")

# Grafico operazioni
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(signals['Price'], label='Prezzo', alpha=0.6)
ax.scatter(signals.index[signals['Buy_Signal']], signals['Price'][signals['Buy_Signal']], marker='^', color='green', label='Buy')
ax.scatter(signals.index[signals['Sell_Signal']], signals['Price'][signals['Sell_Signal']], marker='v', color='red', label='Sell')
ax.set_title(f"Operazioni Buy/Sell - {ticker}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Download CSV
csv = signals[['Price', 'RSI', 'Buy_Signal', 'Sell_Signal', 'Cumulative_Strategy_Return']].to_csv().encode('utf-8')
st.download_button("Scarica segnali (CSV)", csv, file_name=f"{ticker}_signals.csv", mime="text/csv")

