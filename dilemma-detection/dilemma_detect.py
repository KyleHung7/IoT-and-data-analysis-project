import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque

# =========================
# 參數設定
# =========================
video_path = r"D:\AIOTPaper\dilemma\video.mp4"

# Dilemma Zone 公式參數
delta1 = 1.5   # 反應時間 s
delta2 = 1.5   # 反應時間修正 s
tau = 3.5      # 黃燈時間 s
a1 = 3.5       # 最大減速度 m/s²
a2 = 3.0       # 減速度 m/s²
W = 3.5        # 車道寬 m
L = 4.5        # 車長 m

# 影像與透視變換參數
lane_width = 3.5
num_lanes = 3
fps = 30

# --- [新增] 追蹤與過濾參數 ---
# 速度平滑化窗口大小 (幀)
SMOOTHING_WINDOW = 5
# 判定為「已停止」的速度閾值 (m/s)
STOPPED_SPEED_THRESHOLD = 0.5
# 判定為「在停止線前」的距離閾值 (m)
STOP_LINE_PROXIMITY = 3.0
# 追蹤對象的最大存活幀數 (如果一個ID在MAX_AGE幀內都沒更新，就刪除)
MAX_AGE = 15
# 靜態物體過濾：追蹤超過N幀後開始判斷
STATIC_CHECK_FRAME_THRESHOLD = 20
# 靜態物體過濾：在N幀內總位移小於此值(m)，則視為靜態
STATIC_DISPLACEMENT_THRESHOLD = 0.5

# 原始檢測過濾參數
min_bbox_area = 1500
aspect_ratio_range = (0.5, 3.0) # 放寬長寬比上限，以適應不同角度的車輛

# =========================
# 輔助函數 (已修正)
# =========================
def calculate_smoothed_speed(track, fps, window_size):
    """使用滑動窗口計算平滑速度"""
    # track 是一個 deque 物件
    if len(track) < 2:
        return 0.0
    
    # --- [修正] ---
    # 將 deque 轉換為 list 以便進行切片操作
    track_list = list(track)
    
    # 取最近的幾筆數據
    points_to_use = track_list[-window_size:]
    
    # 如果數據不足以進行平滑平均，則退回計算瞬時速度
    if len(points_to_use) < 2:
        dist_m = np.linalg.norm(track_list[-1] - track_list[-2]) / 100.0
        return dist_m * fps

    # 計算窗口內每兩點間的速度，然後平均
    speeds = []
    for i in range(1, len(points_to_use)):
        dist_m = np.linalg.norm(points_to_use[i] - points_to_use[i-1]) / 100.0
        speed_mps = dist_m * fps
        speeds.append(speed_mps)
    
    return np.mean(speeds) if speeds else 0.0

# =========================
# 1. 讀影片第一幀，滑鼠選角點
# =========================
pts_src = []
def click_event(event, x, y, flags, param):
    global pts_src
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_src) < 4:
        pts_src.append([x, y])
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Frame", frame)
        print(f"選取點 {len(pts_src)}: ({x}, {y})")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("❌ 無法讀影片")
    exit()

# 讓使用者可以縮放視窗
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", click_event)
print("請用滑鼠點擊四個角點 (停止線左/右, 畫面底端左/右)，點完按任意鍵繼續")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(pts_src) != 4:
    print("❌ 角點選擇不足四個，程式終止。")
    exit()

pts_src = np.array(pts_src, dtype=np.float32)
print("選取的四個角點:", pts_src)

# =========================
# 2. 計算俯視平面矩形
# =========================
width_pixel = np.linalg.norm(pts_src[1] - pts_src[0])
width_m = lane_width * num_lanes
length_pixel = (np.linalg.norm(pts_src[2] - pts_src[0]) + np.linalg.norm(pts_src[3] - pts_src[1])) / 2
scale = width_m / width_pixel
length_m = length_pixel * scale

# 放大100倍以提高精度
pts_dst = np.array([
    [0, 0],
    [width_m * 100, 0],
    [0, length_m * 100],
    [width_m * 100, length_m * 100]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts_src, pts_dst)

stopline_pixel = ((pts_src[0][0] + pts_src[1][0]) / 2, (pts_src[0][1] + pts_src[1][1]) / 2)
stopline_topview_y = cv2.perspectiveTransform(np.array([[stopline_pixel]], dtype=np.float32), M)[0][0][1]

# =========================
# 3. YOLO 偵測與追蹤
# =========================
model = YOLO("yolov8n.pt")
frame_idx = 0
next_vehicle_id = 0
vehicle_tracks = {} # {id: {'track': deque, 'last_seen': int, 'bbox': tuple}}
results_list = []

h, w = frame.shape[:2]
out = cv2.VideoWriter('dilemma_zone_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- [修改] 只偵測常見車輛類別 ---
    # COCO classes: 2:car, 3:motorcycle, 5:bus, 7:truck
    results = model.predict(frame, conf=0.3, classes=[2, 3, 5, 7])
    
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    # 計算俯視圖中的中心點
    centroids_pixel = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in detections]
    if not centroids_pixel:
        centroids_topview = []
    else:
        centroids_topview = cv2.perspectiveTransform(np.array([centroids_pixel], dtype=np.float32), M)[0]

    # --- [重構] 追蹤邏輯 ---
    current_detections = []
    for i, c_topview in enumerate(centroids_topview):
        x1, y1, x2, y2 = detections[i]
        current_detections.append({'centroid': c_topview, 'bbox': (x1, y1, x2, y2)})

    # 匹配
    matched_ids = set()
    if vehicle_tracks:
        unmatched_detections = list(range(len(current_detections)))
        
        track_ids = list(vehicle_tracks.keys())
        last_centroids = [vehicle_tracks[tid]['track'][-1] for tid in track_ids]

        for i, det in enumerate(current_detections):
            dists = [np.linalg.norm(det['centroid'] - lc) for lc in last_centroids]
            if dists:
                min_dist_idx = np.argmin(dists)
                # 增加一個距離閾值，防止匹配到太遠的目標
                if dists[min_dist_idx] < 50 * 100: # 50m in scaled units
                    track_id = track_ids[min_dist_idx]
                    if track_id not in matched_ids:
                        vehicle_tracks[track_id]['track'].append(det['centroid'])
                        vehicle_tracks[track_id]['last_seen'] = frame_idx
                        vehicle_tracks[track_id]['bbox'] = det['bbox']
                        matched_ids.add(track_id)
                        if i in unmatched_detections:
                            unmatched_detections.remove(i)
        
        # 為未匹配的檢測創建新ID
        for i in unmatched_detections:
            vehicle_tracks[next_vehicle_id] = {
                'track': deque([current_detections[i]['centroid']], maxlen=int(fps*2)), # 最多存2秒的軌跡
                'last_seen': frame_idx,
                'bbox': current_detections[i]['bbox']
            }
            next_vehicle_id += 1
    else:
        # 第一幀或無追蹤對象時
        for det in current_detections:
            vehicle_tracks[next_vehicle_id] = {
                'track': deque([det['centroid']], maxlen=int(fps*2)),
                'last_seen': frame_idx,
                'bbox': det['bbox']
            }
            next_vehicle_id += 1

    # --- [新增] 清理舊的/失追的追蹤對象 ---
    dead_tracks = [vid for vid, data in vehicle_tracks.items() if frame_idx - data['last_seen'] > MAX_AGE]
    for vid in dead_tracks:
        del vehicle_tracks[vid]

    # --- [重構] 計算速度、距離、dilemma zone ---
    for vid, data in list(vehicle_tracks.items()): # 使用 list() 避免在迭代時修改字典
        track = data['track']
        bbox = data['bbox']
        
        if len(track) < 2:
            continue

        # 1. 基本過濾 (BBox 面積與長寬比)
        x1, y1, x2, y2 = map(int, bbox)
        bbox_area = (x2 - x1) * (y2 - y1)
        aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
        if bbox_area < min_bbox_area or not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue

        # 2. 計算平滑速度
        v_smooth = calculate_smoothed_speed(track, fps, SMOOTHING_WINDOW)

        # 3. 計算到停止線的距離 (只考慮y軸)
        # 假設車輛從y值大 -> y值小的方向行駛
        d_m = (track[-1][1] - stopline_topview_y) / 100.0
        
        # --- [新增] 過濾已越過停止線的車輛 ---
        if d_m <= 0:
            continue

        # --- [新增] 過濾靜態物體 (如地面箭頭) ---
        if len(track) > STATIC_CHECK_FRAME_THRESHOLD:
            total_displacement = np.linalg.norm(track[-1] - track[0]) / 100.0 # in meters
            if total_displacement < STATIC_DISPLACEMENT_THRESHOLD:
                # 標記為靜態物體並跳過
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(frame, f'ID:{vid} (Static)', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                continue

        # --- [新增] 過濾已在停止線前停下的車輛 ---
        if v_smooth < STOPPED_SPEED_THRESHOLD and d_m < STOP_LINE_PROXIMITY:
            # 標記為已停止並跳過
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'ID:{vid} (Stopped)', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            continue

        # 4. Dilemma Zone 計算 (使用平滑速度)
        v0 = v_smooth # 使用平滑速度進行計算
        Xc = v0 * delta1 + v0**2 / (2 * a1)
        X0 = v0 * tau - 0.5 * a2 * (tau - delta2)**2 - W - L
        is_dilemma = X0 < d_m < Xc

        results_list.append({
            "frame": frame_idx,
            "vehicle_id": vid,
            "speed_m_s": v0,
            "dist_to_stop_m": d_m,
            "X0": X0,
            "Xc": Xc,
            "dilemma_zone": is_dilemma
        })

        # 5. 視覺化
        color = (0, 0, 255) if is_dilemma else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'ID:{vid} V:{v0:.1f}m/s D:{d_m:.1f}m'
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if is_dilemma:
            cv2.putText(frame, 'DILEMMA', (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# =========================
# 4. 輸出 CSV
# =========================
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv("dilemma_zone_results.csv", index=False)
    print("✅ CSV 輸出完成: dilemma_zone_results.csv")
else:
    print("⚠️ 沒有記錄任何數據，未生成 CSV。")

print("✅ 影片輸出完成: dilemma_zone_output.avi")