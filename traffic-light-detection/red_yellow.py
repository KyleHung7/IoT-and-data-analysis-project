import cv2
import numpy as np

# --- 設定 ---
url = "https://jtmctrafficcctv3.gov.taipei/NVR/57e78f0d-74ef-40ce-a32e-b9151e095885/live.m3u8"

# ROI: 只關心紅黃燈
x1, y1, x2, y2 = 668, 53, 680, 64

def detect_red_yellow(frame, x1, y1, x2, y2):
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

    red_on = np.count_nonzero(mask_red) > 2
    yellow_on = False if red_on else np.count_nonzero(mask_yellow) > 2

    return red_on, yellow_on, mask_red, mask_yellow

def main():
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("無法開啟串流")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(100)
            continue

        # --- 紅黃燈偵測 ---
        red_on, yellow_on, mask_red, mask_yellow = detect_red_yellow(frame, x1, y1, x2, y2)

        # 畫 ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

        # 顯示燈號文字
        if red_on:
            status_text = "RED"
            color = (0,0,255)
        elif yellow_on:
            status_text = "YELLOW"
            color = (0,255,255)
        else:
            status_text = "OFF"
            color = (200,200,200)

        cv2.putText(frame, f"Traffic Light: {status_text}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 顯示 Mask 做 debug
        cv2.imshow("Mask Red", mask_red)
        cv2.imshow("Mask Yellow", mask_yellow)
        cv2.imshow("Traffic Light Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 離開
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
