import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque

# =========================
# 參數設定
# =========================
w, h = 960, 540
url = "https://jtmctrafficcctv3.gov.taipei/NVR/57e78f0d-74ef-40ce-a32e-b9151e095885/live.m3u8"

# Dilemma Zone 公式參數
delta1 = 1.5
delta2 = 1.5
tau = 3.5
a1 = 3.5
a2 = 3.0
W = 3.5
L = 4.5

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
# 紅黃燈偵測
# =========================
def detect_red_yellow(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # HSV 範圍
    red_lower1 = np.array([0, 80, 80])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 80, 80])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([20, 80, 80])
    yellow_upper = np.array([35, 255, 255])

    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # 優先判斷紅燈
    red_on = np.count_nonzero(mask_red) > 2
    yellow_on = False
    if not red_on:
        yellow_on = np.count_nonzero(mask_yellow) > 2

    return red_on, yellow_on, mask_red, mask_yellow

# =========================
# 計算平滑速度
# =========================
def calculate_smoothed_speed(track, fps, window_size):
    if len(track) < 2:
        return 0.0
    points = list(track)[-window_size:]
    if len(points) < 2:
        return 0.0
    speeds = [np.linalg.norm(points[i]-points[i-1])*fps for i in range(1,len(points))]
    return float(np.mean(speeds))

# =========================
# 讀影片與選擇 ROI/四個角點
# =========================
cap = cv2.VideoCapture(url)
ret, frame = cap.read()
if not ret:
    print("❌ 無法開啟串流")
    exit()

frame = cv2.resize(frame, (w, h))
frame_copy = frame.copy()

# 1. 選擇紅黃燈 ROI
roi = cv2.selectROI("Select Traffic Light ROI", frame_copy, showCrosshair=True)
cv2.destroyWindow("Select Traffic Light ROI")
roi_coords = tuple(map(int, roi))  # (x, y, w, h)
x1, y1, w_roi, h_roi = roi_coords
x2, y2 = x1+w_roi, y1+h_roi

# 2. 選擇 Dilemma Zone 四個角點
pts_src = []

def click_event(event, x, y, flags, param):
    global pts_src, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_src)<4:
        pts_src.append([x, y])
        cv2.circle(frame_copy, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Select 4 corners", frame_copy)
        print(f"Selected point {len(pts_src)}: ({x},{y})")

frame_copy2 = frame.copy()
cv2.namedWindow("Select 4 corners", cv2.WINDOW_NORMAL)
cv2.imshow("Select 4 corners", frame_copy2)
cv2.setMouseCallback("Select 4 corners", click_event)
print("請依序點選停止線左/右、畫面底端左/右，點完按任意鍵繼續")
cv2.waitKey(0)
cv2.destroyWindow("Select 4 corners")

if len(pts_src)!=4:
    print("❌ 角點不足四個，程式終止")
    exit()
pts_src = np.array(pts_src, dtype=np.float32)

# 計算俯視矩形
width_m = lane_width*num_lanes
length_pixel = (np.linalg.norm(pts_src[2]-pts_src[0]) + np.linalg.norm(pts_src[3]-pts_src[1]))/2
scale = width_m / np.linalg.norm(pts_src[1]-pts_src[0])
length_m = length_pixel*scale

pts_dst = np.array([
    [0,0],
    [width_m,0],
    [0,length_m],
    [width_m,length_m]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts_src, pts_dst)

stopline_pixel = ((pts_src[0][0]+pts_src[1][0])/2, (pts_src[0][1]+pts_src[1][1])/2)
stopline_topview = cv2.perspectiveTransform(np.array([[stopline_pixel]], dtype=np.float32), M)[0][0]
stopline_topview_y = stopline_topview[1]

# =========================
# YOLO 偵測與追蹤
# =========================
model = YOLO("yolov8n.pt")
frame_idx = 0
next_vehicle_id = 0
vehicle_tracks = {}
results_list = []

out = cv2.VideoWriter('dilemma_zone_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

yellow_phase = False  # 只紀錄黃燈到紅燈之間

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))
    frame_disp = frame.copy()

    # 紅黃燈判斷
    red_on, yellow_on, mask_red, mask_yellow = detect_red_yellow(frame, (x1, y1, x2, y2))

    if yellow_on and not red_on:
        yellow_phase = True
    elif red_on:
        yellow_phase = False  # 黃轉紅結束

    # 顯示紅黃燈
    if red_on:
        status_text = "RED"
        color = (0,0,255)
    elif yellow_on:
        status_text = "YELLOW"
        color = (0,255,255)
    else:
        status_text = "GREEN"
        color = (200,200,200)
    cv2.putText(frame_disp, f"Traffic Light: {status_text}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame_disp, (x1,y1),(x2,y2), color, 2)

    # YOLO 偵測
    results = model.predict(frame, conf=0.3, classes=[2,3,5,7])
    detections = results[0].boxes.xyxy.cpu().numpy()
    centroids_pixel = [((x1+x2)/2,(y1+y2)/2) for x1,y1,x2,y2 in detections]

    centroids_topview = cv2.perspectiveTransform(np.array([centroids_pixel], dtype=np.float32), M)[0] if centroids_pixel else []

    current_detections = []
    for i,c_top in enumerate(centroids_topview):
        current_detections.append({'centroid': c_top, 'bbox': detections[i]})

    # 追蹤
    matched_ids = set()
    if vehicle_tracks:
        unmatched = list(range(len(current_detections)))
        track_ids = list(vehicle_tracks.keys())
        last_centroids = [np.array(vehicle_tracks[tid]['track'][-1]) for tid in track_ids]
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
                        if i in unmatched: unmatched.remove(i)
        for i in unmatched:
            vehicle_tracks[next_vehicle_id] = {'track': deque([current_detections[i]['centroid']], maxlen=int(fps*2)),
                                               'last_seen': frame_idx,
                                               'bbox': current_detections[i]['bbox']}
            next_vehicle_id += 1
    else:
        for det in current_detections:
            vehicle_tracks[next_vehicle_id] = {'track': deque([det['centroid']], maxlen=int(fps*2)),
                                               'last_seen': frame_idx,
                                               'bbox': det['bbox']}
            next_vehicle_id += 1

    dead = [vid for vid,data in vehicle_tracks.items() if frame_idx - data['last_seen'] > MAX_AGE]
    for vid in dead:
        del vehicle_tracks[vid]

    # 紀錄只在黃燈到紅燈之間的車輛
    if yellow_phase:
        for vid,data in vehicle_tracks.items():
            track = data['track']
            bbox = data['bbox']
            if len(track)<2: continue
            v_smooth = calculate_smoothed_speed(track,fps,SMOOTHING_WINDOW)
            v_kmh = v_smooth*3.6
            d_m = track[-1][1]-stopline_topview_y
            if d_m<=0: continue

            # 篩選車輛
            x1b,y1b,x2b,y2b = map(int,bbox)
            bbox_area = (x2b-x1b)*(y2b-y1b)
            aspect_ratio = (x2b-x1b)/(y2b-y1b+1e-6)
            if bbox_area<min_bbox_area or not (aspect_ratio_range[0]<=aspect_ratio<=aspect_ratio_range[1]):
                continue

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
                "dilemma_zone": is_dilemma,
                "traffic_light": status_text
            })

            color_box = (0,0,255) if is_dilemma else (0,255,0)
            cv2.rectangle(frame_disp,(x1b,y1b),(x2b,y2b),color_box,2)
            label = f'ID:{vid} V:{v_kmh:.1f} D:{d_m:.1f} {status_text}'
            cv2.putText(frame_disp,label,(x1b,y1b-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_box,1)

    out.write(frame_disp)
    cv2.imshow("Output", frame_disp)
    if cv2.waitKey(1)&0xFF==27: break
    frame_idx +=1

cap.release()
out.release()
cv2.destroyAllWindows()

# =========================
# 輸出 CSV
# =========================
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv("dilemma_zone_results.csv",index=False)
    print("✅ CSV 輸出完成")
else:
    print("⚠️ 沒有記錄任何數據")
print("✅ 影片輸出完成")
