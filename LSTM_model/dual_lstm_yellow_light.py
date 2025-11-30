# dual_lstm_yellow_light.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# 1️⃣ 讀取資料
# ==========================
df = pd.read_csv("speed_log_labeled.csv")

# 填補可能的 NaN
df[["speed_ms", "distance_to_stop_line", "traffic_density"]] = df[["speed_ms", "distance_to_stop_line", "traffic_density"]].fillna(0)
df["tracker_id"] = df["tracker_id"].fillna(0).astype(int)

features = ["speed_ms", "distance_to_stop_line", "traffic_density"]
target_speed = "speed_ms"
target_decision = "go_stop_label"

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

seq_len = 20

# ==========================
# 2️⃣ 分車建立序列
# ==========================
X_seq, y_speed_seq, y_decision_seq = [], [], []

for tracker in df["tracker_id"].unique():
    df_car = df[df["tracker_id"] == tracker].reset_index(drop=True)
    for i in range(len(df_car) - seq_len):
        X_seq.append(df_car[features].iloc[i:i+seq_len].values)
        y_speed_seq.append(df_car[target_speed].iloc[i+seq_len])
        y_decision_seq.append(df_car[target_decision].iloc[i+seq_len])

X_seq = np.array(X_seq)
y_speed_seq = np.array(y_speed_seq)
y_decision_seq = np.array(y_decision_seq)

print("X shape:", X_seq.shape)
print("y_speed shape:", y_speed_seq.shape)
print("y_decision shape:", y_decision_seq.shape)

# ==========================
# 3️⃣ 建立 LSTM 速度預測模型
# ==========================
model_speed = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # 預測下一瞬間速度
])

model_speed.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mae"]
)

model_speed.summary()

# ==========================
# 4️⃣ 訓練速度模型
# ==========================
es_speed = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history_speed = model_speed.fit(
    X_seq, y_speed_seq,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es_speed]
)

# ==========================
# 5️⃣ 依速度預測計算 Go / Stop
# ==========================
YELLOW_TIME = 3
MAX_DECEL = 6
TTC_THRESHOLD = 1.5

def predict_go_stop(X_input, model_speed):
    pred_speed = model_speed.predict(X_input)
    last_distance = X_input[:, -1, 1]  # 最後一幀距離
    last_ttc = X_input[:, -1, 2]       # 最後一幀 TTC

    go_stop_pred = []

    for v, d, ttc in zip(pred_speed.flatten(), last_distance, last_ttc):
        if v < 2:
            go_stop_pred.append(0)
            continue
        d_brake = (v * v) / (2 * MAX_DECEL)
        can_stop = d_brake < d
        t_pass = d / max(v, 0.1)
        can_go = t_pass < YELLOW_TIME
        if ttc < TTC_THRESHOLD:
            go_stop_pred.append(0)
            continue
        go_stop_pred.append(1 if can_go else 0)

    return np.array(go_stop_pred)

y_pred_decision = predict_go_stop(X_seq, model_speed)
print("前20筆預測 Go/Stop:", y_pred_decision[:20])

# ==========================
# 6️⃣ 儲存模型與結果
# ==========================
model_speed.save("yellow_light_speed_lstm.h5")
np.save("predicted_go_stop.npy", y_pred_decision)
print("✔ 模型與 Go/Stop 預測已儲存")
