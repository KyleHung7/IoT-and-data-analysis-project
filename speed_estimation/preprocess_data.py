import json
import pandas as pd
import numpy as np

# 1. 定義目標 CSV 的欄位 (根據你提供的範例)
TARGET_COLUMNS = [
    "frame_index", "tracker_id", "vehicle_type", "class_id", "x", "y", 
    "distance", "time_s", "speed_kmh", "speed_ms", "traffic_light_status", 
    "yellow_light", "distance_to_stop_line", "distance_to_front_vehicle", 
    "traffic_density", "ttc", "yellow_light_decision"
]

def process_jsonl(file_path, decision_label):
    """
    讀取 jsonl 檔案並轉換為符合目標格式的 list of dicts
    file_path: 檔案路徑
    decision_label: 'GO' 或 'STOP' (對應 yellow_light_decision)
    """
    processed_rows = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 為了計算相對時間，我們記錄第一筆數據的時間戳記
            start_t_ms = None
            
            for line in lines:
                if not line.strip(): continue # 跳過空行
                data = json.loads(line)
                
                # 初始化時間基準
                if start_t_ms is None:
                    start_t_ms = data.get('t_ms', 0)

                # --- 欄位映射邏輯 ---
                
                # 1. 速度轉換
                v_mps = data.get('v_mps', 0.0)
                speed_kmh = v_mps * 3.6
                
                # 2. 時間計算 (轉成秒，相對時間)
                current_t_ms = data.get('t_ms', 0)
                time_s = (current_t_ms - start_t_ms) / 1000.0
                
                # 3. 建立單行資料
                row = {
                    "frame_index": data.get('idx'),        # 使用 idx 作為 frame_index
                    "tracker_id": 1,                       # 假設這是單一車輛視角，固定為 1
                    "vehicle_type": "car",                 # 固定
                    "class_id": 2,                         # 固定
                    "x": 4,                                # 範例值 (因為 sensor data 沒有車道橫向座標)
                    "y": 28,                               # 範例值
                    "distance": 0,                         # 範例中多為0或空，這裡設為0
                    "time_s": round(time_s, 3),            # 保留三位小數
                    "speed_kmh": round(speed_kmh, 2),
                    "speed_ms": v_mps,
                    "traffic_light_status": "YELLOW",      # 固定
                    "yellow_light": True,                  # 固定
                    
                    # 注意：sensor data 沒有「距離停止線」的資料
                    # 這裡先填入範例中的 28.0，若你有經緯度計算邏輯需在此修改
                    "distance_to_stop_line": 28.0,         
                    
                    "distance_to_front_vehicle": None,     # Sensor data 無此資料
                    "traffic_density": 1,                  # 範例值
                    "ttc": None,                           # Sensor data 無此資料
                    "yellow_light_decision": decision_label # 關鍵標籤：GO 或 STOP
                }
                
                processed_rows.append(row)
                
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {file_path}")
        return []

    return processed_rows

# 2. 處理兩個檔案
# 假設你的檔案名稱如下，請根據實際情況修改
go_data = process_jsonl('sensor_yellow_go.jsonl', 'GO')
stop_data = process_jsonl('sensor_yellow_stop.jsonl', 'STOP')

# 3. 合併資料
all_data = go_data + stop_data

# 4. 轉為 DataFrame
df = pd.DataFrame(all_data)

# 確保欄位順序正確 (依照 TARGET_COLUMNS)
df = df[TARGET_COLUMNS]

# 5. 輸出 CSV
output_filename = 'merged_sensor_data.csv'
df.to_csv(output_filename, index=False)

print(f"轉換完成！共處理 {len(df)} 筆資料。")
print(f"檔案已儲存為: {output_filename}")

# 顯示前幾筆看看結果
print(df[['frame_index', 'speed_kmh', 'yellow_light_decision']].head())