import cv2

url = "https://jtmctrafficcctv3.gov.taipei/NVR/57e78f0d-74ef-40ce-a32e-b9151e095885/live.m3u8"  # 你的 m3u8


cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

if not cap.isOpened():
    raise RuntimeError("打不開串流，請確認 OpenCV 是否包含 FFmpeg，或改用下方 PyAV/FFmpeg 管線。")

while True:
    ok, frame = cap.read()
    if not ok:
        print("讀取失敗，嘗試重新連線…")
        cap.release()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)

    cv2.imshow("live", frame)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
