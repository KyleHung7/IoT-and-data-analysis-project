import re
import json
import math
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------- 使用說明 --------------
# python video_sensor_sync.py VID_20250207_142355_ms12345.mp4 sensor_log.jsonl
#
# 影片檔名需要包含 msXXXXXX 這段，代表這部影片第一個 frame 對應的 t_ms_start
# JSONL 每行格式類似：
# {"idx": 123, "t_ms": 13000, "ts":"20250207_142355", ... , "ax_g":..., "ay_g":..., "az_g":..., "dhtTempC":..., "dhtHum":...}
# --------------------------------------


def parse_ms_from_filename(video_path: Path) -> int:
    """
    從檔名抓出 msXXXXX → 回傳 t_ms_start (int)
    例：VID_20250207_142355_ms12345.mp4 → 12345
    """
    m = re.search(r"_ms(\d+)", video_path.stem)
    if not m:
        raise ValueError(f"檔名中找不到 '_msNNNN' 這種格式：{video_path.name}")
    return int(m.group(1))


def load_sensor_jsonl(jsonl_path: Path, t_ms_start: int):
    """
    讀取 jsonl，轉成 numpy 陣列
    回傳:
        t_sec  : shape (N,) 相對影片開始時間（秒）
        ax, ay, az, temp, hum ...（你可以再擴充）
    """
    t_list = []
    ax_list, ay_list, az_list = [], [], []
    temp_list, hum_list = [], []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "t_ms" not in obj:
                continue

            t_ms = obj["t_ms"]
            # 影片時間 = t_ms - 影片第一個 frame 的 t_ms_start
            t_sec = (t_ms - t_ms_start) / 1000.0

            # 只收錄影片開始之後的資料
            if t_sec < 0:
                continue

            t_list.append(t_sec)

            def get_num(key):
                v = obj.get(key, None)
                if v is None:
                    return math.nan
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return math.nan

            ax_list.append(get_num("ax_g"))
            ay_list.append(get_num("ay_g"))
            az_list.append(get_num("az_g"))
            temp_list.append(get_num("dhtTempC"))
            hum_list.append(get_num("dhtHum"))

    if not t_list:
        raise RuntimeError("JSONL 裡沒有有效的資料（t_ms）。")

    t_sec = np.array(t_list, dtype=float)
    ax = np.array(ax_list, dtype=float)
    ay = np.array(ay_list, dtype=float)
    az = np.array(az_list, dtype=float)
    temp = np.array(temp_list, dtype=float)
    hum = np.array(hum_list, dtype=float)

    return t_sec, ax, ay, az, temp, hum


def main(video_file, jsonl_file):
    video_path = Path(video_file)
    jsonl_path = Path(jsonl_file)

    if not video_path.is_file():
        print("找不到影片檔：", video_path)
        return
    if not jsonl_path.is_file():
        print("找不到 JSONL 檔：", jsonl_path)
        return

    # 1) 從檔名擷取 t_ms_start
    t_ms_start = parse_ms_from_filename(video_path)
    print("影片第一幀 t_ms_start =", t_ms_start)

    # 2) 讀 jsonl → 時間與感測器陣列
    print("讀取 JSONL...")
    t_sec, ax, ay, az, temp, hum = load_sensor_jsonl(jsonl_path, t_ms_start)
    print(f"共讀入 {len(t_sec)} 筆感測器資料")

    # 3) 開啟影片
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("無法開啟影片：", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    print("影片 FPS =", fps)

    # 4) 建立 Matplotlib 圖（互動模式）
    plt.ion()
    fig, (ax_acc, ax_env) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 三軸加速度
    line_ax, = ax_acc.plot([], [], label="ax (g)")
    line_ay, = ax_acc.plot([], [], label="ay (g)")
    line_az, = ax_acc.plot([], [], label="az (g)")
    ax_acc.set_ylabel("Acceleration (g)")
    ax_acc.legend(loc="upper right")
    ax_acc.grid(True)

    # 溫度 / 濕度
    line_temp, = ax_env.plot([], [], label="Temp (C)", color="tab:red")
    ax_env.set_ylabel("Temp / Hum")
    ax_env.set_xlabel("Time (s)")
    ax_env.grid(True)

    ax2b = ax_env.twinx()
    line_hum, = ax2b.plot([], [], label="Hum (%)", color="tab:blue")

    # 用來決定 y 範圍
    def finite_minmax(arr):
        m = np.nanmin(arr)
        M = np.nanmax(arr)
        if not np.isfinite(m) or not np.isfinite(M) or m == M:
            return -1, 1
        return m, M

    # 讓兩個 y 軸的 legend 都能顯示
    lines = [line_temp, line_hum]
    labels = [l.get_label() for l in lines]
    ax_env.legend(lines, labels, loc="upper right")

    plt.tight_layout()

    # 5) 主迴圈：讀 frame + 更新 plot
    sensor_idx = 0  # 指到下一筆還沒用過的感測器資料
    n_sensor = len(t_sec)

    print("開始播放，按 'q' 關閉視窗")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("影片播放結束")
            break

        # 影片目前時間（秒）
        cur_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 找到「時間 <= cur_sec」的所有感測器資料
        # 這裡用指標線性往前走，比每次 np.searchsorted 較省
        while sensor_idx < n_sensor and t_sec[sensor_idx] <= cur_sec:
            sensor_idx += 1

        if sensor_idx > 0:
            # 用 0 ~ sensor_idx-1 的資料來畫
            t_used = t_sec[:sensor_idx]
            ax_used = ax[:sensor_idx]
            ay_used = ay[:sensor_idx]
            az_used = az[:sensor_idx]
            temp_used = temp[:sensor_idx]
            hum_used = hum[:sensor_idx]

            # 更新線條
            line_ax.set_data(t_used, ax_used)
            line_ay.set_data(t_used, ay_used)
            line_az.set_data(t_used, az_used)

            line_temp.set_data(t_used, temp_used)
            line_hum.set_data(t_used, hum_used)

            # 更新範圍
            ax_acc.set_xlim(0, max(t_used))
            y_min, y_max = finite_minmax(
                np.concatenate([ax_used, ay_used, az_used])
            )
            ax_acc.set_ylim(y_min - 0.1, y_max + 0.1)

            if np.any(np.isfinite(temp_used)):
                tmin, tmax = finite_minmax(temp_used)
                ax_env.set_ylim(tmin - 1, tmax + 1)

            if np.any(np.isfinite(hum_used)):
                hmin, hmax = finite_minmax(hum_used)
                ax2b.set_ylim(hmin - 5, hmax + 5)

        # 更新圖
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 顯示影片
        cv2.imshow("Video", frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法:")
        print("  python video_sensor_sync.py <video.mp4> <sensor_log.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
