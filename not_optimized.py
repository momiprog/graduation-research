import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import datetime as dt
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Zeros
import random
import os
import tensorflow as tf


os.environ["PYTHONHASHSEED"] = "1234"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)


# 1. データ取得
ticker = "7203.T"  # トヨタ自動車
start_date = "2018-01-01"
end_date = "2024-12-31"

print("データ取得中")
data = yf.download(ticker, start=start_date, end=end_date)

# 終値を使用
close_data = data["Close"].values.reshape(-1, 1)

# 2. 前処理
# 正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# 時系列データの作成（過去60日分 → 翌日）
def create_sequences(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i - window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 訓練・テスト分割
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. モデル構築
def build_lstm_model(n_units1=256, n_units2=256, dropout1=0.2, dropout2=0.2, lr=0.001):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),   # ✅ 警告を防ぐ方法
        LSTM(int(n_units1), return_sequences=True,
             kernel_initializer=GlorotUniform(seed=1234),
            recurrent_initializer=Orthogonal(seed=1234),
            bias_initializer=Zeros()),
        Dropout(float(dropout1), seed=1234),
        LSTM(int(n_units2), return_sequences=False,
             kernel_initializer=GlorotUniform(seed=1234),
            recurrent_initializer=Orthogonal(seed=1234),
            bias_initializer=Zeros()),
        Dropout(float(dropout2), seed=1234),
        Dense(25, activation='relu',kernel_initializer=GlorotUniform(seed=1234)),
        Dense(1,kernel_initializer=GlorotUniform(seed=1234))
    ])
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss='mean_squared_error')
    return model

# # 4. モデル学習 & 可視化付きハイパーパラメータ探索

# param_list = [
#     {"n_units1": 64, "n_units2": 64, "dropout1": 0.2, "dropout2": 0.2, "lr": 0.001},
#     {"n_units1": 64, "n_units2": 64, "dropout1": 0.5, "dropout2": 0.025, "lr": 0.005},
#     {"n_units1": 64,  "n_units2": 64,  "dropout1": 0.2, "dropout2": 0.2, "lr": 0.001},
#     {"n_units1": 64, "n_units2": 64, "dropout1": 0.3, "dropout2": 0.3, "lr": 0.0005},
# ]

# results = []

# for i, params in enumerate(param_list):
#     print("\n" + "="*60)
#     print(f"モデル {i+1}/{len(param_list)} 実行中")
#     print("params =", params)
#     print("="*60)

#     model = build_lstm_model(**params)

#     history = model.fit(
#         X_train, y_train,
#         epochs=20,
#         batch_size=32,
#         validation_data=(X_test, y_test),
#         verbose=0
#     )

#     # ===== 予測 =====
#     predicted = model.predict(X_test)
#     predicted_price = scaler.inverse_transform(predicted)
#     actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

#     # ===== MSE =====
#     mse = mean_squared_error(actual_price, predicted_price)
#     print(f"MSE: {mse:.2f}")

#     # ===== 可視化 =====
#     plt.figure(figsize=(12, 6))
#     plt.plot(actual_price, label="Actual Price")
#     plt.plot(predicted_price, label="Predicted Price")
#     plt.title(
#         f"Model {i+1} | "
#         f"units1={params['n_units1']} "
#         f"units2={params['n_units2']} "
#         f"drop1={params['dropout1']} "
#         f"drop2={params['dropout2']} "
#         f"lr={params['lr']} | "
#         f"MSE={mse:.2f}"
#     )
#     plt.xlabel("Days")
#     plt.ylabel("Stock Price (Yen)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # ===== 結果保存 =====
#     result = params.copy()
#     result["MSE"] = mse
#     results.append(result)

# # ============================================================
# # 結果一覧
# # ============================================================

# df_results = pd.DataFrame(results).sort_values("MSE")
# print("\nハイパーパラメータ探索結果（MSE昇順）")
# print(df_results)

# best_params = df_results.iloc[0].to_dict()
# print("\n最良パラメータ:", best_params)


# 4. モデル学習（複数パラメータ同時探索版）


param_list = [
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.013, "dropout2": 0.325, "lr": 0.001},
]





results = []

for i, params in enumerate(param_list):
    print("\n" + "="*60)
    print(f"モデル {i+1}/{len(param_list)} 実行中: {params}")
    print("="*60)

    model = build_lstm_model(**params)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )

    predicted = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted)
    actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(actual_price, predicted_price)

    print(f"MSE: {mse:.2f}")

    result = params.copy()
    result["MSE"] = mse
    results.append(result)

# 結果を DataFrame にまとめる
df_results = pd.DataFrame(results).sort_values("MSE")
print("\nハイパーパラメータ探索結果")
print(df_results)

# 最良モデル
best_params = df_results.iloc[0].to_dict()
print("\n最良パラメータ:", best_params)

# # 5. 予測と評価
# predicted = best_model.predict(X_test)
# predicted_price = scaler.inverse_transform(predicted)
# actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# # 評価指標
# mse = mean_squared_error(actual_price, predicted_price)
# # directional_accuracy = np.mean(
# #     np.sign(np.diff(actual_price, axis=0)) == np.sign(np.diff(predicted_price, axis=0))
# # )

# print(f"MSE: {mse:.2f}")
# # print(f"方向一致率: {directional_accuracy * 100:.2f}%")

# # 6. 可視化
# plt.figure(figsize=(12, 6))
# plt.plot(actual_price, label="true stock price", color="black")
# plt.plot(predicted_price, label="predicted stock price", color="red")
# plt.title(f"{ticker} LSTM")
# plt.xlabel("days")
# plt.ylabel("stock price(yen)")
# plt.legend()
# plt.show()

# # 7. 未来予測（直近60日→翌日）
# last_60_days = scaled_data[-window_size:]
# X_future = np.reshape(last_60_days, (1, window_size, 1))
# future_price_scaled = best_model.predict(X_future)
# future_price = scaler.inverse_transform(future_price_scaled)

# print(f"翌日の予測株価: {future_price[0][0]:.2f} 円")
