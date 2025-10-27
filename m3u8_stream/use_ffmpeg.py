import subprocess as sp
import numpy as np
import cv2
import shlex

w, h = 960, 540  # 你要的處理解析度

url = "https://jtmctrafficcctv3.gov.taipei/NVR/57e78f0d-74ef-40ce-a32e-b9151e095885/live.m3u8"  # 你的 m3u8


# For nvidia gpu
# args = [
#     "ffmpeg", "-hide_banner", "-loglevel", "warning",
#     "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2",
#     "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "1M", "-analyzeduration", "2000000",
#     "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "8",
#     "-i", url,
#     "-vf", f"hwdownload,format=nv12,scale={w}:{h},format=bgr24",
#     "-f", "rawvideo", "-"
# ]

# For intel gpu
args = [
    "ffmpeg", "-hide_banner", "-loglevel", "warning",
    "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2",
    "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "1M", "-analyzeduration", "2000000",
    "-hwaccel", "qsv", "-hwaccel_output_format", "qsv", "-c:v", "h264_qsv",  # HEVC 改 hevc_qsv
    "-i", url,
    # 先把硬體幀拉回 RAM 並轉成 NV12，再用 CPU 縮放，最後轉成 OpenCV 要的 BGR24
    "-vf", f"hwdownload,format=nv12,scale={w}:{h},format=bgr24",
    "-f", "rawvideo", "-"
]


proc = sp.Popen(args, stdout=sp.PIPE, bufsize=10**8)
frame_size = w*h*3
while True:
    raw = proc.stdout.read(frame_size)
    if len(raw) != frame_size:
        break
    frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 15)

    cv2.imshow("live", frame)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) & 0xFF == 27:
        break
proc.terminate()
cv2.destroyAllWindows()
