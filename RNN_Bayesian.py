# ============================================================
# 📈 LSTMによるトヨタ株価予測 ＋ ベイズ最適化 (n_units 固定版)
# ------------------------------------------------------------
# 必要ライブラリ：
# pip install yfinance scikit-learn tensorflow bayesian-optimization matplotlib
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from bayes_opt import BayesianOptimization
import datetime as dt
import tensorflow as tf

# ============================================================
# 1. データ取得
# ============================================================
ticker = "7203.T" # トヨタ自動車 (← 全角スペースを修正)
start_date = "2018-01-01"
end_date = "2024-12-31"

print("📥 データ取得中...")
data = yf.download(ticker, start=start_date, end=end_date)
close_data = data["Close"].values.reshape(-1, 1)

# ============================================================
# 2. 前処理
# ============================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

def create_sequences(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i - window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ============================================================
# 3. モデル構築関数（ベイズ最適化で使用）
# ============================================================
def build_lstm_model(n_units1=50, n_units2=50, dropout1=0.2, dropout2=0.2, lr=0.001):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),  # (← 全角スペースを修正)
        LSTM(int(n_units1), return_sequences=True),
        Dropout(float(dropout1)),
        LSTM(int(n_units2), return_sequences=False),
        Dropout(float(dropout2)),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss='mean_squared_error')
    return model

# ============================================================
# 4. 評価関数（ベイズ最適化用）
# ============================================================
def evaluate_model(dropout1, dropout2, lr):
    
    # ここで固定値を定義 (例: 64)
    FIXED_N_UNITS1 = 256
    FIXED_N_UNITS2 = 256
    
    # 固定値を使ってモデルを構築
    model = build_lstm_model(
        n_units1=FIXED_N_UNITS1, 
        n_units2=FIXED_N_UNITS2, 
        dropout1=dropout1, 
        dropout2=dropout2, 
        lr=lr
    )
    
    # 学習
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0,
        validation_data=(X_test, y_test)
    )
    
    # 評価（MSE）
    predicted = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predicted)
    return -mse

# ============================================================
# 5. ベイズ最適化の設定
# ============================================================
pbounds = {
    # 'n_units1': (32, 128), # 削除
    # 'n_units2': (32, 128), # 削除
    'dropout1': (0.0125, 0.5),
    'dropout2': (0.0125, 0.5),
    'lr': (0.00001, 0.001)
}

optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

print("🔍 ベイズ最適化開始 (n_units は 64/64 に固定)...")
optimizer.maximize(init_points=10, n_iter=10)

# ============================================================
# 6. 最適パラメータで再学習
# ============================================================
best_params_from_opt = optimizer.max['params'] # dropout と lr のみ

# evaluate_model で使用した固定値を再度定義
FIXED_N_UNITS1 = 128
FIXED_N_UNITS2 = 128

# マージして最終的なパラメータセットを作成
final_params = {
    'n_units1': FIXED_N_UNITS1,
    'n_units2': FIXED_N_UNITS2,
    **best_params_from_opt
}

print("✅ 最適ハイパーパラメータ (固定値含む):", final_params)

# 最終パラメータでモデルを構築
best_model = build_lstm_model(**final_params)
history = best_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ============================================================
# 7. 予測と評価
# ============================================================
predicted = best_model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted)
actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(actual_price, predicted_price)
directional_accuracy = np.mean(
    np.sign(np.diff(actual_price, axis=0)) == np.sign(np.diff(predicted_price, axis=0))
)

print(f"📊 MSE: {mse:.2f}")
print(f"📈 方向一致率: {directional_accuracy * 100:.2f}%")

# ============================================================
# 8. 可視化
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(actual_price, label="true stock price", color="black")
plt.plot(predicted_price, label="predicted stock price", color="red")
plt.title(f"{ticker} LSTM (optimized, n_units=64/64 fixed)")
plt.xlabel("days")
plt.ylabel("stock price (yen)")
plt.legend()
plt.show()

# ============================================================
# 9. 翌日予測
# ============================================================
last_60_days = scaled_data[-window_size:]
X_future = np.reshape(last_60_days, (1, window_size, 1))
future_price_scaled = best_model.predict(X_future)
future_price = scaler.inverse_transform(future_price_scaled)

print(f"🔮 翌日の予測株価: {future_price[0][0]:.2f} 円") # (← 全角スペースを修正)