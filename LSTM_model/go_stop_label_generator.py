# go_stop_label_generator.py
import pandas as pd
import numpy as np

YELLOW_TIME = 3        # 黃燈秒數
MAX_DECEL = 6          # 最大減速度 (m/s^2)
TTC_THRESHOLD = 1.5    # 安全 TTC 門檻 (秒)

df = pd.read_csv("speed_log.csv")

# 填補可能的 NaN
df["speed_ms"] = df["speed_ms"].fillna(0)
df["distance_to_stop_line"] = df["distance_to_stop_line"].fillna(999)
df["ttc"] = df["ttc"].fillna(999)

go_stop_labels = []

for i, row in df.iterrows():
    v = row["speed_ms"]                         # 當前速度 (m/s)
    d = row["distance_to_stop_line"]            # 距離停車線距離 (m)
    ttc = row["ttc"]                            

    # ==========================
    # 1. 若速度極低 → 默認停下來
    # ==========================
    if v < 2:
        go_stop_labels.append(0)
        continue

    # ==========================
    # 2. 剎停距離判斷 (STOP)
    #    d_brake = v² / 2a
    # ==========================
    d_brake = (v * v) / (2 * MAX_DECEL)
    can_stop = d_brake < d

    if can_stop:
        go_stop_labels.append(0)
        continue

    # ==========================
    # 3. 是否能在黃燈剩餘時間通過 (GO)
    #    t_pass = d / v
    # ==========================
    t_pass = d / v
    can_go = t_pass < YELLOW_TIME

    # ==========================
    # 4. 安全性判斷 (TTC 過低 → STOP)
    # ==========================
    if ttc < TTC_THRESHOLD:
        go_stop_labels.append(0)
        continue

    # ==========================
    # 最終決策
    # ==========================
    if can_go:
        go_stop_labels.append(1)
    else:
        go_stop_labels.append(0)

df["go_stop_label"] = go_stop_labels
df.to_csv("speed_log_labeled.csv", index=False)

print("✔ 已完成：speed_log_labeled.csv")
print(df[["speed_ms", "distance_to_stop_line", "go_stop_label"]].head())
