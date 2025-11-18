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
delta1 = 1.5
delta2 = 1.5
tau = 3.5
a1 = 3.5
a2 = 3.0
W = 3.5
L = 4.5

# 影像與透視變換參數
lane_width = 3.5
num_lanes = 3
fps = 30

# 追蹤與過濾參數
SMOOTHING_WINDOW = 5
STOPPED_SPEED_THRESHOLD = 1.8  # km/h -> 0.5 m/s
STOP_LINE_PROXIMITY = 3.0
MAX_AGE = 15
STATIC_CHECK_FRAME_THRESHOLD = 20
STATIC_DISPLACEMENT_THRESHOLD = 0.5

min_bbox_area = 1500
aspect_ratio_range = (0.5, 3.0)

# =========================
# 輔助函數
# =========================
def calculate_smoothed_speed(track, fps, window_size):
    """計算平滑速度 (m/s)"""
    if len(track) < 2:
        return 0.0
    
    points = list(track)[-window_size:]
    if len(points) < 2:
        dist = np.linalg.norm(points[-1] - points[-2])
        return dist * fps

    speeds = []
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i-1])
        speed = dist * fps
        speeds.append(speed)
    return float(np.mean(speeds))

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

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", frame.shape[1], frame.shape[0])
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

# 目標矩形直接使用米
pts_dst = np.array([
    [0, 0],
    [width_m, 0],
    [0, length_m],
    [width_m, length_m]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts_src, pts_dst)

stopline_pixel = ((pts_src[0][0]+pts_src[1][0])/2, (pts_src[0][1]+pts_src[1][1])/2)
stopline_topview_y = cv2.perspectiveTransform(
    np.array([[stopline_pixel]], dtype=np.float32), M
)[0][0][1]

# =========================
# 3. YOLO 偵測與追蹤
# =========================
model = YOLO("yolov8n.pt")
frame_idx = 0
next_vehicle_id = 0
vehicle_tracks = {}
results_list = []

h, w = frame.shape[:2]
out = cv2.VideoWriter('dilemma_zone_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.3, classes=[2,3,5,7])
    detections = results[0].boxes.xyxy.cpu().numpy()
    centroids_pixel = [((x1+x2)/2,(y1+y2)/2) for x1,y1,x2,y2 in detections]
    centroids_topview = cv2.perspectiveTransform(
        np.array([centroids_pixel], dtype=np.float32), M
    )[0] if centroids_pixel else []

    current_detections = []
    for i, c_topview in enumerate(centroids_topview):
        x1, y1, x2, y2 = detections[i]
        current_detections.append({'centroid': c_topview, 'bbox': (x1,y1,x2,y2)})

    matched_ids = set()
    if vehicle_tracks:
        unmatched = list(range(len(current_detections)))
        track_ids = list(vehicle_tracks.keys())
        last_centroids = [vehicle_tracks[tid]['track'][-1] for tid in track_ids]

        for i, det in enumerate(current_detections):
            dists = [np.linalg.norm(det['centroid']-lc) for lc in last_centroids]
            if dists:
                min_idx = np.argmin(dists)
                if dists[min_idx] < 50.0:
                    tid = track_ids[min_idx]
                    if tid not in matched_ids:
                        vehicle_tracks[tid]['track'].append(det['centroid'])
                        vehicle_tracks[tid]['last_seen'] = frame_idx
                        vehicle_tracks[tid]['bbox'] = det['bbox']
                        matched_ids.add(tid)
                        if i in unmatched:
                            unmatched.remove(i)
        for i in unmatched:
            vehicle_tracks[next_vehicle_id] = {
                'track': deque([current_detections[i]['centroid']], maxlen=int(fps*2)),
                'last_seen': frame_idx,
                'bbox': current_detections[i]['bbox']
            }
            next_vehicle_id += 1
    else:
        for det in current_detections:
            vehicle_tracks[next_vehicle_id] = {
                'track': deque([det['centroid']], maxlen=int(fps*2)),
                'last_seen': frame_idx,
                'bbox': det['bbox']
            }
            next_vehicle_id += 1

    dead = [vid for vid,data in vehicle_tracks.items() if frame_idx - data['last_seen'] > MAX_AGE]
    for vid in dead:
        del vehicle_tracks[vid]

    for vid,data in list(vehicle_tracks.items()):
        track = data['track']
        bbox = data['bbox']
        if len(track)<2:
            continue

        x1,y1,x2,y2 = map(int,bbox)
        bbox_area = (x2-x1)*(y2-y1)
        aspect_ratio = (x2-x1)/(y2-y1+1e-6)
        if bbox_area<min_bbox_area or not (aspect_ratio_range[0]<=aspect_ratio<=aspect_ratio_range[1]):
            continue

        v_smooth = calculate_smoothed_speed(track,fps,SMOOTHING_WINDOW)
        v_kmh = v_smooth * 3.6  # m/s -> km/h
        d_m = track[-1][1] - stopline_topview_y
        if d_m<=0: continue

        if len(track)>STATIC_CHECK_FRAME_THRESHOLD:
            total_disp = np.linalg.norm(track[-1]-track[0])
            if total_disp<STATIC_DISPLACEMENT_THRESHOLD:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),1)
                cv2.putText(frame,f'ID:{vid} (Static)',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
                continue

        if v_kmh<STOPPED_SPEED_THRESHOLD and d_m<STOP_LINE_PROXIMITY:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(frame,f'ID:{vid} (Stopped)',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            continue

        # Dilemma Zone 計算
        Xc = v_smooth*delta1 + v_smooth**2/(2*a1)
        X0 = v_smooth*tau - 0.5*a2*(tau-delta2)**2 - W - L
        is_dilemma = X0<d_m<Xc

        results_list.append({
            "frame": frame_idx,
            "vehicle_id": vid,
            "speed_kmh": v_kmh,
            "dist_to_stop_m": d_m,
            "X0": X0,
            "Xc": Xc,
            "dilemma_zone": is_dilemma
        })

        color = (0,0,255) if is_dilemma else (0,255,0)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        label = f'ID:{vid} V:{v_kmh:.1f} km/h D:{d_m:.1f} m'
        cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        if is_dilemma:
            cv2.putText(frame,'DILEMMA',(x1,y2+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    out.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'): break
    frame_idx+=1

cap.release()
out.release()
cv2.destroyAllWindows()

# =========================
# 4. 輸出 CSV
# =========================
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv("dilemma_zone_results.csv",index=False)
    print("✅ CSV 輸出完成")
else:
    print("⚠️ 沒有記錄任何數據")

print("✅ 影片輸出完成")
