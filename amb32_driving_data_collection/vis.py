import serial
import time
import collections
import math
import threading

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from flask import Flask, jsonify, render_template_string

# ========= 串口設定 =========
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

# ========= 資料緩衝區設定 =========
MAX_POINTS = 300

pitch_buf = collections.deque(maxlen=MAX_POINTS)
roll_buf = collections.deque(maxlen=MAX_POINTS)

ax_buf = collections.deque(maxlen=MAX_POINTS)
ay_buf = collections.deque(maxlen=MAX_POINTS)
az_buf = collections.deque(maxlen=MAX_POINTS)

temp_buf = collections.deque(maxlen=MAX_POINTS)
hum_buf = collections.deque(maxlen=MAX_POINTS)
acc_mag_buf = collections.deque(maxlen=MAX_POINTS)

time_buf = collections.deque(maxlen=MAX_POINTS)

# ========= GPS =========
last_lat = float("nan")
last_lon = float("nan")
last_speed = float("nan")   # m/s
last_accel = float("nan")   # m/s^2

# ========= 連接串口 =========
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
time.sleep(2)
print("Serial connected on", SERIAL_PORT)
ser.reset_input_buffer()

# ========= Flask Web =========
app = Flask(__name__)


@app.route("/gps")
def gps_api():
    """回傳最新 GPS 座標與速度（km/h）"""
    global last_lat, last_lon, last_speed, last_accel
    if math.isnan(last_lat) or math.isnan(last_lon):
        fix = False
    else:
        fix = True

    if math.isnan(last_speed):
        v_kmh = None
    else:
        v_kmh = last_speed * 3.6

    if math.isnan(last_accel):
        a_ms2 = None
    else:
        a_ms2 = last_accel

    return jsonify({
        "fix": fix,
        "lat": last_lat,
        "lon": last_lon,
        "speed_kmh": v_kmh,
        "accel_ms2": a_ms2,
    })


# Leaflet 地圖頁，一次載入，之後只更新 marker，不整頁重整
LEAFLET_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GPS Live Map</title>
  <!-- Leaflet CSS -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body {
      height: 100%;
      margin: 0;
      padding: 0;
    }
    #map {
      width: 100%;
      height: 100vh;
    }
    #info {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 1000;
      background: rgba(255,255,255,0.8);
      padding: 6px 10px;
      border-radius: 4px;
      font-family: sans-serif;
      font-size: 14px;
    }
  </style>
  <!-- Leaflet JS -->
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin="">
  </script>
</head>
<body>
  <div id="info">GPS: loading...</div>
  <div id="map"></div>

  <script>
    // 初始中心（先放台北），等有 GPS fix 再移過去
    let map = L.map('map').setView([25.0330, 121.5654], 16);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    let marker = null;
    let hasEverFixed = false;

    async function updateGPS() {
      try {
        const res = await fetch('/gps');
        const data = await res.json();

        const infoDiv = document.getElementById('info');

        if (!data.fix || data.lat === null || data.lon === null) {
          if (!hasEverFixed) {
            infoDiv.textContent = 'GPS: no fix yet';
          } else {
            infoDiv.textContent = 'GPS: lost fix';
          }
          return;
        }

        const lat = data.lat;
        const lon = data.lon;
        const v = data.speed_kmh;
        const a = data.accel_ms2;

        hasEverFixed = true;

        let text = `Lat=${lat.toFixed(6)}, Lon=${lon.toFixed(6)}`;
        if (v !== null) {
          text += `, v=${v.toFixed(2)} km/h`;
        } else {
          text += `, v=-- km/h`;
        }
        if (a !== null) {
          text += `, a=${a.toFixed(2)} m/s²`;
        } else {
          text += `, a=-- m/s²`;
        }

        infoDiv.textContent = text;

        const latlng = [lat, lon];

        if (!marker) {
          marker = L.marker(latlng).addTo(map);
        } else {
          marker.setLatLng(latlng);
        }

        marker.bindPopup(text);

        // 可選：地圖跟著移動，可以視需要註解掉
        map.setView(latlng);

      } catch (e) {
        document.getElementById('info').textContent = 'Error: ' + e;
      }
    }

    // 每秒更新一次
    setInterval(updateGPS, 1000);
    updateGPS();
  </script>
</body>
</html>
"""


@app.route("/")
def index_page():
    return render_template_string(LEAFLET_PAGE)


def run_web():
    # 重要：use_reloader=False，避免 Flask 開第二個進程干擾
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ========= 建立圖表 =========
plt.style.use("default")
fig = plt.figure(figsize=(14, 10))

# (1) Pitch / Roll
ax1 = fig.add_subplot(3, 2, 1)
line_pitch, = ax1.plot([], [], label="Pitch (deg)")
line_roll,  = ax1.plot([], [], label="Roll (deg)")
ax1.set_ylabel("Angle (deg)")
ax1.legend()
ax1.grid(True)

# (2) 三軸加速度折線圖
ax_acc = fig.add_subplot(3, 2, 2)
line_ax, = ax_acc.plot([], [], label="ax (g)")
line_ay, = ax_acc.plot([], [], label="ay (g)")
line_az, = ax_acc.plot([], [], label="az (g)")
ax_acc.set_ylabel("Acceleration (g)")
ax_acc.set_xlabel("Sample Index")
ax_acc.legend()
ax_acc.grid(True)

# (3) Temp / Hum
ax2 = fig.add_subplot(3, 2, 3)
line_temp, = ax2.plot([], [], label="Temp (°C)", color="red")
ax2.set_ylabel("Temp (°C)", color="red")
ax2.grid(True)

ax2b = ax2.twinx()
line_hum, = ax2b.plot([], [], label="Humidity (%)", color="blue")
ax2b.set_ylabel("Humidity (%)")

# (4) |a| Histogram
ax3 = fig.add_subplot(3, 2, 4)
hist_bins = 20
ax3.set_xlabel("|a| (g)")
ax3.set_ylabel("Count")
ax3.set_title("Acceleration Magnitude Histogram")

plt.tight_layout()


def parse_line(line):
    """
    格式:
    DATA,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mpuTempC,pitch,roll,dhtTempC,dhtHum,lat,lon,speed,accel
    """
    line = line.strip()
    if not line.startswith("DATA,"):
        return None

    parts = line.split(",")
    if len(parts) != 16:
        return None

    try:
        ax_g = float(parts[1])
        ay_g = float(parts[2])
        az_g = float(parts[3])

        pitch = float(parts[8])
        roll = float(parts[9])
        dhtT = float(parts[10])
        dhtH = float(parts[11])

        lat = float(parts[12])
        lon = float(parts[13])
        speed = float(parts[14])   # m/s，可能是 nan
        accel = float(parts[15])   # m/s^2，可能是 nan

        acc_mag = math.sqrt(ax_g**2 + ay_g**2 + az_g**2)

        return {
            "ax": ax_g,
            "ay": ay_g,
            "az": az_g,
            "pitch": pitch,
            "roll": roll,
            "dhtT": dhtT,
            "dhtH": dhtH,
            "acc_mag": acc_mag,
            "lat": lat,
            "lon": lon,
            "speed": speed,
            "accel": accel,
        }

    except ValueError:
        return None


def update(frame):
    global last_lat, last_lon, last_speed, last_accel

    # 多讀幾行避免 lag
    for _ in range(20):
        raw = ser.readline().decode("utf-8", errors="ignore")
        if not raw:
            break

        data = parse_line(raw)
        if data is None:
            continue

        idx = 0 if len(time_buf) == 0 else time_buf[-1] + 1
        time_buf.append(idx)

        ax_buf.append(data["ax"])
        ay_buf.append(data["ay"])
        az_buf.append(data["az"])
        pitch_buf.append(data["pitch"])
        roll_buf.append(data["roll"])
        temp_buf.append(data["dhtT"])
        hum_buf.append(data["dhtH"])
        acc_mag_buf.append(data["acc_mag"])

        last_lat = data["lat"]
        last_lon = data["lon"]
        last_speed = data["speed"]
        last_accel = data["accel"]

    if len(time_buf) == 0:
        return

    x = list(time_buf)

    # (1) Pitch / Roll
    line_pitch.set_data(x, list(pitch_buf))
    line_roll.set_data(x, list(roll_buf))
    ax1.set_xlim(min(x), max(x))
    ax1.set_ylim(
        min(min(pitch_buf), min(roll_buf)) - 5,
        max(max(pitch_buf), max(roll_buf)) + 5
    )

    # (2) 三軸加速度
    line_ax.set_data(x, list(ax_buf))
    line_ay.set_data(x, list(ay_buf))
    line_az.set_data(x, list(az_buf))
    ax_acc.set_xlim(min(x), max(x))
    ax_acc.set_ylim(
        min(min(ax_buf), min(ay_buf), min(az_buf)) - 0.2,
        max(max(ax_buf), max(ay_buf), max(az_buf)) + 0.2
    )

    # (3) Temp / Hum
    line_temp.set_data(x, list(temp_buf))
    line_hum.set_data(x, list(hum_buf))
    ax2.set_xlim(min(x), max(x))
    ax2.set_ylim(min(temp_buf) - 1, max(temp_buf) + 1)
    ax2b.set_ylim(min(hum_buf) - 5, max(hum_buf) + 5)

    # (4) |a| Histogram
    ax3.cla()
    ax3.hist(list(acc_mag_buf), bins=hist_bins, range=(0, 3))
    ax3.set_title("Acceleration Magnitude Histogram")
    ax3.set_xlabel("|a| (g)")
    ax3.set_ylabel("Count")

    # 顯示 GPS & 速度 / 加速度 在圖視窗標題
    if not math.isnan(last_lat) and not math.isnan(last_lon):
        if not math.isnan(last_speed):
            v_kmh = last_speed * 3.6
            if not math.isnan(last_accel):
                fig.suptitle(
                    f"Lat={last_lat:.6f}, Lon={last_lon:.6f}, v={v_kmh:.2f} km/h, a={last_accel:.2f} m/s²"
                )
            else:
                fig.suptitle(
                    f"Lat={last_lat:.6f}, Lon={last_lon:.6f}, v={v_kmh:.2f} km/h, a=-- m/s²"
                )
        else:
            fig.suptitle(
                f"Lat={last_lat:.6f}, Lon={last_lon:.6f}, v=-- km/h, a=-- m/s²"
            )
    else:
        fig.suptitle("GPS: no fix")

    return


# ========= 啟動 Flask server（平行執行） =========
web_thread = threading.Thread(target=run_web, daemon=True)
web_thread.start()

# ========= 啟動畫圖 =========
ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
plt.show()

ser.close()
