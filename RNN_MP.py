#LSTMによるトヨタ株価予測 ＋ ベイズ最適化

import GPy
import math
import itertools
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
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Zeros
import random
import os

os.environ["PYTHONHASHSEED"] = "1234"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

# def rbf_kernel(X1, X2, length_scale=1):
#     """RBFカーネル関数"""
#     sq_dists = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1.dot(X2.T)
#     return np.exp(-sq_dists /(2 * length_scale**2))

# def mutual_information(K_A, sigma2):
#     # I(y_A; f_A) = 0.5 * log det(I + sigma^-2 K_A)
#     n = K_A.shape[0]
#     return 0.5 * np.linalg.slogdet(np.eye(n) + (1/sigma2) * K_A)[1]

# def greedy_max_info_gain(X, kernel, sigma2, t):
#     n = len(X)
#     if t > n:
#         raise ValueError(f"t={t} はサンプル数 n={n} より大きいので選べない")

#     A = []

#     for _ in range(t):
#         best_gain = -np.inf
#         best_idx = None

#         for i in range(n):
#             if i in A:
#                 continue

#             A_new = A + [i]
#             X_A = X[A_new]                  #先に部分抽出
#             K_A = kernel(X_A, X_A)          #部分だけ計算
#             gain = mutual_information(K_A, sigma2)

#             if gain > best_gain:
#                 best_gain = gain
#                 best_idx = i

#         if best_idx is None:
#             raise RuntimeError("候補点が尽きたが t 点の選択を要求されている")

#         A.append(best_idx)

#     return A

# 3. モデル構築関数（ベイズ最適化で使用）
def build_lstm_model(dropout1=0.2, dropout2=0.2, lr=0.001):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),   #警告を防ぐ
        LSTM(int(256), return_sequences=True,
            kernel_initializer=GlorotUniform(seed=1234),
            recurrent_initializer=Orthogonal(seed=1234),
            bias_initializer=Zeros()),
        Dropout(float(dropout1), seed=1234),
        LSTM(int(256), return_sequences=False,
            kernel_initializer=GlorotUniform(seed=1234),
            recurrent_initializer=Orthogonal(seed=1234),
            bias_initializer=Zeros(),),
        Dropout(float(dropout2), seed=1234),
        Dense(25, activation='relu',kernel_initializer=GlorotUniform(seed=1234)),
        Dense(1,kernel_initializer=GlorotUniform(seed=1234))
    ])
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss='mean_squared_error')
    return model

# # 1. データ取得
# ticker = "7203.T"  # トヨタ自動車
# start_date = "2018-01-01"
# end_date = "2024-12-31"

# print("データ取得中")
# data = yf.download(ticker, start=start_date, end=end_date)

# close_data = data["Close"].values.reshape(-1, 1)

# # 2. 前処理（LSTM用・株価の正規化）

# scaler = MinMaxScaler(feature_range=(0, 1))   # ✅ 株価専用
# scaled_data = scaler.fit_transform(close_data)

# def create_sequences(dataset, window_size=60):
#     X, y = [], []
#     for i in range(window_size, len(dataset)):
#         X.append(dataset[i - window_size:i, 0])
#         y.append(dataset[i, 0])
#     return np.array(X), np.array(y)

# window_size = 60
# X, y = create_sequences(scaled_data, window_size)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

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
train_data = close_data.loc["2018-01-01":"2023-10-03"]

# テスト入力用：2023–2024（2024予測のために必要）
test_input_data = close_data.loc["2023-10-04":"2024-12-31"]

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

# 4. 評価関数（ベイズ最適化用）
def evaluate_model(dropout1, dropout2, lr):
    model = build_lstm_model(dropout1, dropout2, lr)

    # 学習
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        verbose=0,
        validation_data=(X_test, y_test)
    )

    # 評価（MSE）
    predicted = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predicted)
    return -mse  # ベイズ最適化では最大化なのでマイナス

print("top-kアルゴリズム開始")
# optimizer.maximize(init_points=5, n_iter=10)

n_train = 10             # 事前分布のサンプリング関数
d = 3                    # 入力次元の数
variance = 0.001          #MSEの分散
T = 20                   #サンプリング回数
k = 10                   #トップk
values = [round(i * 0.004, 4) for i in range(1, 101)]
param_grid = [values,values,
 [round(i * 0.000008, 6) for i in range(1, 101)]]
x_all = list(itertools.product(*param_grid))
x_all = np.array(x_all)     # NumPy配列に変換
x_all = x_all.astype(np.float32)
x_input = np.array(x_all)[
    np.random.choice(len(x_all), size=n_train, replace=False)
]  #事前分布のx, x_allからランダムに選択する
y_output = []  # 事前分布の y（= -MSE）
for i in range(n_train):
    print("事前入力",i+1,"回目",x_input[i])
    new_y = evaluate_model(x_input[i][0], x_input[i][1], x_input[i][2])
    y_output.append(new_y)

#ndarray化
y_output = np.array(y_output).reshape(-1, 1)

#y だけ MinMax 正規化
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_output_minmax = y_scaler.fit_transform(y_output)


for t in range(T):
    kern = GPy.kern.RBF(input_dim=d)
    model = GPy.models.GPRegression(x_input, y_output_minmax, kern)
    model.optimize_restarts(num_restarts=5, verbose=False)
    # 予測（x_all: (n_train, d)）
    mu, var = model.predict(x_all)          # (n_train,1), (n_train,1)

    # 結合[x1,...,xd, mu, var]
    x_mu_var = np.hstack((x_all, mu, var))  # shape = (n_train, d+2)
    mu_col_idx = d                          # mu は列 index d、var は d+1

    # mu で降順ソートして top-k を得る
    sortedx_mu_var = x_mu_var[np.argsort(x_mu_var[:, mu_col_idx])[::-1]]
    top_k = sortedx_mu_var[:k]
    top_k_c = sortedx_mu_var[k:]

    # top_k と not-top_k の lower/upper を作る（数値で保持）
    top_k_lu = []     # 各要素: [x1...xd, lower, upper]
    # A = greedy_max_info_gain(x_all, rbf_kernel, variance, t-1)
    # #A に対応する点だけを先に抜き出す
    # X_A = x_all[A]
    # #必要な部分カーネルだけを直接計算
    # K_A = rbf_kernel(X_A, X_A)
    # gamma = mutual_information(K_A, variance)
    # print("近似最大情報利得 γ_{t-1} =", gamma)
    B = 2
    C = 0.5
    if t > 1:
      beta = (B + math.sqrt(variance) * math.sqrt(2 * C *(math.log(t+1))**(d+1) + 1 + math.log(1/0.0001)))**2
    else:
      beta = B
    sum_mu =  0
    sum_beta_val = 0
    for row in top_k:
        x_vec = row[:d].astype(float)
        mu_val = float(row[mu_col_idx])
        var_val = max(float(row[mu_col_idx + 1]), 1e-12)
        beta_val = math.sqrt(beta) * math.sqrt(var_val)
        sum_mu += mu_val
        sum_beta_val += beta_val
        lower = mu_val - beta_val
        upper = mu_val + beta_val
        top_k_lu.append(list(x_vec) + [lower, upper])
    # print("ave_beta:" , sum_beta_val / k)
    # print("ave_mu:" , sum_mu / k)

    top_k_c_lu = []
    sum_mu =  0
    sum_beta_val = 0
    for row in top_k_c:
        x_vec = row[:d].astype(float)
        mu_val = float(row[mu_col_idx])
        var_val = max(float(row[mu_col_idx + 1]), 1e-12)
        beta_val = math.sqrt(beta) * math.sqrt(var_val)
        sum_mu += mu_val
        sum_beta_val += beta_val
        lower = mu_val - beta_val
        upper = mu_val + beta_val
        top_k_c_lu.append(list(x_vec) + [lower, upper])

    # print("ave_beta:" , sum_beta_val / k)
    # print("ave_mu:" , sum_mu / k)

    upper_confidence_bounds = []
    for i_row in top_k_lu:
        for j_row in top_k_c_lu:
            lower_i = i_row[d]
            upper_i = i_row[d+1]
            lower_j = j_row[d]
            upper_j = j_row[d+1]
            # overlap 定義（>=0）
            overlap = max(0.0, upper_j - lower_i)
            row = i_row[:d] + [lower_i, upper_i] + j_row[:d] + [lower_j, upper_j] + [overlap]
            upper_confidence_bounds.append(row)

    # フォールバック：top_k_c が空、または上の計算で何もできなかった場合
    if len(upper_confidence_bounds) == 0:
        # とりあえずランダムサンプリング（または top_k の中から選ぶ）
        sampling_x = np.atleast_2d(x_all[np.random.randint(0, len(x_all))])
    else:
        # print(upper_confidence_bounds[:,-1])
        upper_confidence_bounds = np.array(upper_confidence_bounds)
        # print(upper_confidence_bounds[:, -1])  # 最後の列を全部取り出す
        ucb_array = np.array(upper_confidence_bounds, dtype=float)  # shape (M, 2*d+5)
        overlap_col = -1
        max_idx = int(np.argmax(ucb_array[:, overlap_col]))
        best_row = ucb_array[max_idx]                               #最もupper confidence bound of the regretが大きかった組合せ
        # print("ucd_array:",ucb_array[max_idx])
        # インデックス計算（汎用）
        # best_row layout: [x_i(0..d-1), lower_i, upper_i, x_j(0..d-1), lower_j, upper_j, overlap]
        # lower_i index = d
        lower_i_idx = d
        upper_i_idx = d + 1
        xj_start = d + 2
        xj_end = xj_start + d
        lower_j_idx = xj_end
        upper_j_idx = xj_end + 1
        overlap_idx = xj_end + 2

        upper_i = best_row[upper_i_idx]
        upper_j = best_row[upper_j_idx]
        lower_i = best_row[lower_i_idx]
        lower_j = best_row[lower_j_idx]

        if upper_i > upper_j:
            xi = best_row[0:d]
            sampling_x = np.atleast_2d(xi)   # shape (1,d)
        else:
            xj = best_row[xj_start:xj_end]
            sampling_x = np.atleast_2d(xj)   # shape (1,d)

    # new_y を取得（func は (n,d) を期待
    print("サンプリング",t+1,"回目:",sampling_x)
    # new_y = func(sampling_x) + np.random.normal(0, variance, (sampling_x.shape[0], 1))
    new_y = evaluate_model(sampling_x[0,0], sampling_x[0,1], sampling_x[0,2])
    new_y = np.array([[new_y]])   # shape=(1,1)にする

    # 形を合わせて追加
    x_input = np.vstack((x_input, sampling_x))
    # print("x_input:",x_input)
    y_output = np.vstack((y_output, new_y))
    # print("y_output:",y_output)
    y_output_minmax = y_scaler.fit_transform(y_output)

kern = GPy.kern.RBF(input_dim=d)
model = GPy.models.GPRegression(x_input, y_output_minmax, kern)
model.optimize_restarts(num_restarts=5, verbose=False)
# 予測（x_all: (n_train, d)）
mu, var = model.predict(x_all)          # (n_train,1), (n_train,1)

# 結合: 列は [x1,...,xd, mu, var]
x_mu_var = np.hstack((x_all, mu, var))  # shape = (n_train, d+2)
mu_col_idx = d                          # mu は列 index d、var は d+1

# mu で降順ソートして top-k を得る
sortedx_mu_var = x_mu_var[np.argsort(x_mu_var[:, mu_col_idx])[::-1]]
top_k = sortedx_mu_var[:k]
top_k_c = sortedx_mu_var[k:]

# 6. 最適パラメータで再学習
best_params = top_k[:, :-2]
print("最適ハイパーパラメータ:", best_params)

best_param = best_params[0]
param_dict = {
    "dropout1": float(best_param[0]),
    "dropout2": float(best_param[1]),
    "lr": float(best_param[2])
}
best_model = build_lstm_model(**param_dict)
history = best_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 7. 予測と評価
predicted = best_model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted)
actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(actual_price, predicted_price)
directional_accuracy = np.mean(
    np.sign(np.diff(actual_price, axis=0)) == np.sign(np.diff(predicted_price, axis=0))
)

print(f"MSE: {mse:.2f}")
print(f"方向一致率: {directional_accuracy * 100:.2f}%")

# 8. 可視化
plt.figure(figsize=(12, 6))
plt.plot(actual_price, label="true stock price", color="black")
plt.plot(predicted_price, label="predicted stock price", color="red")
plt.title(f"{ticker} LSTM (optimized)")
plt.xlabel("days")
plt.ylabel("stock price (yen)")
plt.legend()
plt.show()

## 9. 翌日予測
#last_60_days = scaled_data[-window_size:]
#X_future = np.reshape(last_60_days, (1, window_size, 1))
#future_price_scaled = best_model.predict(X_future)
#future_price = scaler.inverse_transform(future_price_scaled)

#print(f"翌日の予測株価: {future_price[0][0]:.2f} 円")
