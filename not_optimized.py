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


# # 1. データ取得
# ticker = "7203.T"  # トヨタ自動車
# start_date = "2018-01-01"
# end_date = "2024-12-31"

# print("データ取得中")
# data = yf.download(ticker, start=start_date, end=end_date)

# # 終値を使用
# close_data = data["Close"].values.reshape(-1, 1)

# # 2. 前処理
# # 正規化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(close_data)

# # 時系列データの作成（過去60日分 → 翌日）
# def create_sequences(dataset, window_size=60):
#     X, y = [], []
#     for i in range(window_size, len(dataset)):
#         X.append(dataset[i - window_size:i, 0])
#         y.append(dataset[i, 0])
#     return np.array(X), np.array(y)

# window_size = 60
# X, y = create_sequences(scaled_data, window_size)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # 訓練・テスト分割
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# =========================
# 1. データ取得
# =========================
ticker = "7203.T"  # トヨタ自動車
start_date = "2018-01-01"
end_date = "2024-12-31"

print("データ取得中...")
data = yf.download(ticker, start=start_date, end=end_date)

close_data = data[["Close"]]

# =========================
# 2. データ分割（時系列）
# =========================
# 学習用：2018–2022
train_data = close_data.loc["2018-01-01":"2022-12-31"]

# テスト入力用：2023–2024（2024予測のために必要）
test_input_data = close_data.loc["2023-01-01":"2024-12-31"]

# =========================
# 3. 正規化（学習データのみでfit）
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_input_data)

# =========================
# 4. シーケンス作成関数
# =========================
def create_sequences(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i - window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60

# 学習データ
X_train, y_train = create_sequences(train_scaled, window_size)

# テスト用（2023–2024を含む）
X_test_all, y_test_all = create_sequences(test_scaled, window_size)

# =========================
# 5. 2024年分のみ抽出（評価対象）
# =========================
test_dates = test_input_data.index[window_size:]
mask_2024 = test_dates >= "2024-01-01"

X_test = X_test_all[mask_2024]
y_test = y_test_all[mask_2024]

# =========================
# 6. LSTM用に reshape
# =========================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# =========================
# 7. 確認
# =========================
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

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
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000002},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000004},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000006},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000008},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000010},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000012},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000014},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000016},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000018},
    {"n_units1": 256, "n_units2": 256, "dropout1": 0.002, "dropout2": 0.2, "lr": 0.000020},
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
