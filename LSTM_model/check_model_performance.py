import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ==========================
# 1️⃣ Load labeled CSV
# ==========================
df = pd.read_csv("speed_log_labeled.csv")

# Fill NaNs
df[["speed_ms", "distance_to_stop_line", "traffic_density"]] = df[["speed_ms", "distance_to_stop_line", "traffic_density"]].fillna(0)
df["tracker_id"] = df["tracker_id"].fillna(0).astype(int)

features = ["speed_ms", "distance_to_stop_line", "traffic_density"]
target_speed = "speed_ms"
target_decision = "go_stop_label"

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

seq_len = 20

# ==========================
# 2️⃣ Build sequences (all cars together)
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
# 3️⃣ Load model safely
# ==========================
model_speed = load_model("yellow_light_speed_lstm.h5", compile=False)

# ==========================
# 4️⃣ Predict speeds
# ==========================
pred_speed = model_speed.predict(X_seq).flatten()

# ==========================
# 5️⃣ Compute Go/Stop decisions
# ==========================
YELLOW_TIME = 3
MAX_DECEL = 6
TTC_THRESHOLD = 1.5

# Use distance_to_stop_line as d, traffic_density as proxy for TTC
last_distance = X_seq[:, -1, 1]
last_ttc = X_seq[:, -1, 2]

def compute_go_stop(pred_speed, last_distance, last_ttc):
    go_stop_pred = []
    for v, d, ttc in zip(pred_speed, last_distance, last_ttc):
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

y_pred_decision = compute_go_stop(pred_speed, last_distance, last_ttc)

# ==========================
# 6️⃣ Save predictions to CSV
# ==========================
df_result = pd.DataFrame({
    "predicted_speed": pred_speed,
    "actual_speed": y_speed_seq,
    "predicted_go_stop": y_pred_decision,
    "actual_go_stop": y_decision_seq
})
df_result.to_csv("prediction_results.csv", index=False)
print("✔ Predictions saved to prediction_results.csv")

# ==========================
# 7️⃣ Plot results
# ==========================
plt.figure(figsize=(12,6))
plt.plot(df_result["actual_speed"], label="Actual Speed", alpha=0.7)
plt.plot(df_result["predicted_speed"], label="Predicted Speed", alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("Speed (m/s)")
plt.title("Actual vs Predicted Speed (All Cars)")
plt.legend()
plt.tight_layout()
plt.savefig("speed_prediction.png")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df_result["actual_go_stop"], label="Actual Go/Stop", alpha=0.7, drawstyle='steps-post')
plt.plot(df_result["predicted_go_stop"], label="Predicted Go/Stop", alpha=0.7, drawstyle='steps-post')
plt.xlabel("Sample Index")
plt.ylabel("Go (1) / Stop (0)")
plt.title("Actual vs Predicted Go/Stop Decisions (All Cars)")
plt.legend()
plt.tight_layout()
plt.savefig("go_stop_prediction.png")
plt.show()

print("✔ Plots saved as speed_prediction_full.png and go_stop_prediction_full.png")
