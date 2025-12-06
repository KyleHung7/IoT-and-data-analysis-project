#include <Wire.h>
#include "DHT.h"
#include <math.h>

// ======== WiFi & Multimedia / RTSP / MP4 相關 ========
#include "WiFi.h"
#include "StreamIO.h"
#include "VideoStream.h"
#include "AudioStream.h"
#include "AudioEncoder.h"
#include "MP4Recording.h"
#include "RTSP.h"

// ======== 文件系統 (SD) ========
#include "AmebaFatFS.h"

AmebaFatFS fs;
File sensorLog;
bool fsOK = false;
unsigned long frameIdx = 0;   // sample index

// ================== WiFi 設定 ==================
char ssid[] = "juke";      // TODO: 改成你的 WiFi SSID
char pass[] = "asdfghjkl";  // TODO: 改成你的 WiFi 密碼
int wifiStatus = WL_IDLE_STATUS;
bool wifiConnected = false;

// ---- 連線到 Wi-Fi（有重試上限 / timeout）----
const unsigned long WIFI_TIMEOUT_MS = 20000;  // 最多嘗試 20 秒
unsigned long wifiStart = millis();


// ================== RTSP / Camera / Audio / MP4 設定 ==================
#define CHANNEL 0    // RTSP & MP4 使用的 video channel

VideoSetting config(VIDEO_FHD, CAM_FPS, VIDEO_H264, 0);

// 音訊與錄影
AudioSetting configA(0);   // 8kHz Mono Analog Mic
Audio audio;
AAC aac;
MP4Recording mp4;
RTSP rtsp;

// StreamIO：Audio -> AAC、Camera+AAC -> RTSP+MP4
StreamIO audioStreamer(1, 1);   // 1 Input Audio -> 1 Output AAC
StreamIO avMixStreamer(2, 2);   // 2 Input Video+Audio -> 2 Output RTSP+MP4

// MP4 錄影控制（自己切段，不用 fileCount）
const unsigned long SEGMENT_MS = 60000; // 每段 60 秒，可自行調整
bool recording = false;
unsigned long recordingStartMs = 0;

// ================== 感測器腳位與位址 ==================

// AMB82-mini 預設 Wire 使用 I2C 腳位：12 (SDA), 13 (SCL)

// MPU6050 (I2C)
#define MPU_ADDR    0x68   // AD0 接 GND 時為 0x68

// DHT11 在腳 7
#define DHT_PIN     7
#define DHT_TYPE    DHT11

DHT dht(DHT_PIN, DHT_TYPE);

// 轉換常數 (MPU6050 資料手冊)
const float ACCEL_SENS = 16384.0f;   // +/-2g
const float GYRO_SENS  = 131.0f;     // +/-250deg/s

// 感測器資料
float ax_g = NAN, ay_g = NAN, az_g = NAN;
float gx_dps = NAN, gy_dps = NAN, gz_dps = NAN;
float mpuTempC = NAN;
float pitch_deg = NAN, roll_deg = NAN;
float dhtTempC = NAN, dhtHum = NAN;

// GPS 相關
float gpsLatDec = NAN;
float gpsLonDec = NAN;
bool  gpsHasFix = false;

// GPS 速度 / 加速度
float gpsSpeed_mps   = NAN;
float gpsAccel_mps2  = NAN;
float lastSpeed_mps  = NAN;
unsigned long lastSpeedUpdateMs = 0;

// ================== 時間標記（由 GPS GPRMC 的日期+時間計算） ==================
char g_timeTag[20] = "";     // 例如 "20251130_153045"
bool g_timeTagValid = false; // 是否已有有效時間標記

// ================== 函式宣告 ==================
bool readMPU();
bool readDHT();
void pollGPS();
bool parseGPRMC(const String &sentence);
bool parseGPGGA(const String &sentence);
bool parseLatLon(const String &latField, const String &nsField,
                 const String &lonField, const String &ewField,
                 float &lat, float &lon);

void initFSAndOpenLog();
void logFrameToSD(bool okMPU, bool okDHT, unsigned long t_ms);
void handleMP4Recording(unsigned long frameTimeMs);

// ================== setup =====================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // ---- 連線到 Wi-Fi ----
  while (wifiStatus != WL_CONNECTED && (millis() - wifiStart) < WIFI_TIMEOUT_MS) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);
    wifiStatus = WiFi.begin(ssid, pass);
    delay(2000);  // 每 2 秒重試一次
  }

  if (wifiStatus == WL_CONNECTED) {
    wifiConnected = true;
    Serial.print("WiFi connected, IP = ");
    Serial.println(WiFi.localIP());
  } else {
    wifiConnected = false;
    Serial.println("WiFi connect failed, continue without WiFi / RTSP");
  }

  // ---- DHT ----
  dht.begin();
  delay(2000);  // DHT 上電穩定時間

  // ---- I2C ----
  Wire.begin();
  Wire.setClock(100000);

  // ---- MPU6050 喚醒 ----
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1
  Wire.write(0x00);  // 清除 sleep bit
  Wire.endTransmission();

  // ---- GPS: Serial1 (21/22) ----
  Serial1.begin(9600);

  // ---- 初始化 SD + JSON log ----
  initFSAndOpenLog();

  // 給 Python 用的 header
  Serial.println("#TYPE,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mpuTempC,pitch,roll,dhtTempC,dhtHum,lat,lon,speed,accel");

  // ========== Camera + Audio + RTSP + MP4 初始化 ==========

  // 1) 設定 Camera video channel
  Camera.configVideoChannel(CHANNEL, config);
  Camera.videoInit();

  // 2) 設定 Audio（麥克風）
  audio.configAudio(configA);
  audio.begin();

  // 3) 設定 AAC 編碼器
  aac.configAudio(configA);
  aac.begin();

  // 4) 設定 MP4 錄影（解析度 / 音訊格式）
  mp4.configVideo(config);
  mp4.configAudio(configA, CODEC_AAC);
  // 不使用 setRecordingDuration / setRecordingFileCount，改成自己用 begin()/end() 切段

  // 5) 設定 RTSP（畫面 + 聲音）
  if (wifiConnected) {
    rtsp.configVideo(config);
    rtsp.configAudio(configA, CODEC_AAC);
    rtsp.begin();
    Serial.println("[RTSP] RTSP server started");
  } else {
    Serial.println("[RTSP] skip RTSP because WiFi not connected");
  }

  // 6) 建立 Audio StreamIO：Audio -> AAC
  audioStreamer.registerInput(audio);
  audioStreamer.registerOutput(aac);
  if (audioStreamer.begin() != 0) {
    Serial.println("[Audio] StreamIO link start failed");
  }

  // 7) 建立 AV Mix StreamIO：Camera + AAC -> RTSP + MP4
  avMixStreamer.registerInput1(Camera.getStream(CHANNEL)); // Video in
  avMixStreamer.registerInput2(aac);                       // Audio in
  avMixStreamer.registerOutput1(rtsp);                     // RTSP out
  avMixStreamer.registerOutput2(mp4);                      // MP4 out
  if (avMixStreamer.begin() != 0) {
    Serial.println("[AV] StreamIO link start failed");
  }

  // 8) 開啟 Camera channel
  Camera.channelBegin(CHANNEL);

  Serial.println("[RTSP] RTSP + MP4 pipeline setup done, waiting for time tag to start recording...");
  Serial.println("VLC 播放 RTSP 範例：rtsp://<板子IP>/live  (實際路徑以 SDK 範例為準)");
}

// ================== loop ======================
void loop() {
  // 先更新 GPS（也會順便更新時間標記）
  pollGPS();

  // 取得這個 frame 的時間戳（之後整段影片都以第一個 frame 的這個值命名）
  unsigned long frameTimeMs = millis();

  // 讀 MPU（約 30Hz）
  bool okMPU = readMPU();

  // DHT 約 1Hz
  static unsigned long lastDhtMs = 0;
  unsigned long now = frameTimeMs;
  bool okDHT = false;
  if (now - lastDhtMs >= 1000) {
    okDHT = readDHT();
    lastDhtMs = now;
  }

  frameIdx++;

  // ====== Serial CSV for Python ======
  if (okMPU || okDHT || gpsHasFix) {
    Serial.print("DATA,");
    Serial.print(ax_g, 4);      Serial.print(",");
    Serial.print(ay_g, 4);      Serial.print(",");
    Serial.print(az_g, 4);      Serial.print(",");
    Serial.print(gx_dps, 4);    Serial.print(",");
    Serial.print(gy_dps, 4);    Serial.print(",");
    Serial.print(gz_dps, 4);    Serial.print(",");
    Serial.print(mpuTempC, 2);  Serial.print(",");
    Serial.print(pitch_deg, 2); Serial.print(",");
    Serial.print(roll_deg, 2);  Serial.print(",");
    Serial.print(dhtTempC, 2);  Serial.print(",");
    Serial.print(dhtHum, 2);    Serial.print(",");
    Serial.print(gpsLatDec, 6); Serial.print(",");
    Serial.print(gpsLonDec, 6); Serial.print(",");
    Serial.print(gpsSpeed_mps, 3);  Serial.print(",");
    Serial.print(gpsAccel_mps2, 3);
    Serial.println();
  }

  // ====== 寫一筆 JSON 到 SD (使用這個 frame 的 t_ms) ======
  logFrameToSD(okMPU, okDHT, frameTimeMs);

  // ====== 控制 MP4 錄影：用「這個 frame」的 t_ms 當作可能的開檔時間 ======
  handleMP4Recording(frameTimeMs);

  // 讓 loop 頻率接近 30fps（跟 1080p30 視訊大致對齊）
  delay(33);
}

// ================== 檔案系統初始化 ==================
void initFSAndOpenLog() {
  if (!fs.begin()) {
    Serial.println("[FS] 初始化失敗，請檢查 SD 卡");
    fsOK = false;
    return;
  }

  String path = String(fs.getRootPath()) + "sensor_log.jsonl";
  sensorLog = fs.open(path);
  if (!sensorLog) {
    Serial.println("[FS] 開啟 log 檔失敗");
    fsOK = false;
    return;
  }

  fsOK = true;
  Serial.print("[FS] log 檔: ");
  Serial.println(path);
}

// ================== JSON 寫檔 ==================
void logFrameToSD(bool okMPU, bool okDHT, unsigned long t_ms) {
  if (!fsOK || !sensorLog) return;

  sensorLog.print("{\"idx\":");
  sensorLog.print(frameIdx);
  sensorLog.print(",\"t_ms\":");
  sensorLog.print(t_ms);

  // 時間標記字串（由 GPS 來的 UTC 時間）
  sensorLog.print(",\"ts\":\"");
  if (g_timeTagValid) {
    sensorLog.print(g_timeTag);
  } else {
    sensorLog.print("");
  }
  sensorLog.print("\"");

  sensorLog.print(",\"mpu_ok\":");
  sensorLog.print(okMPU ? "true" : "false");
  sensorLog.print(",\"dht_ok\":");
  sensorLog.print(okDHT ? "true" : "false");
  sensorLog.print(",\"gps_fix\":");
  sensorLog.print(gpsHasFix ? "true" : "false");

  #define JNUM(key, cond, val, prec)        \
    sensorLog.print(",\"" key "\":");       \
    if (cond && !isnan(val)) sensorLog.print(val, prec); \
    else sensorLog.print("null")

  JNUM("ax_g", okMPU, ax_g, 4);
  JNUM("ay_g", okMPU, ay_g, 4);
  JNUM("az_g", okMPU, az_g, 4);
  JNUM("gx_dps", okMPU, gx_dps, 4);
  JNUM("gy_dps", okMPU, gy_dps, 4);
  JNUM("gz_dps", okMPU, gz_dps, 4);
  JNUM("mpuTempC", okMPU, mpuTempC, 2);
  JNUM("pitch_deg", okMPU, pitch_deg, 2);
  JNUM("roll_deg", okMPU, roll_deg, 2);

  JNUM("dhtTempC", okDHT, dhtTempC, 2);
  JNUM("dhtHum",  okDHT, dhtHum,   2);

  JNUM("lat",  gpsHasFix, gpsLatDec, 6);
  JNUM("lon",  gpsHasFix, gpsLonDec, 6);
  JNUM("v_mps", gpsHasFix, gpsSpeed_mps, 3);
  JNUM("a_mps2", gpsHasFix, gpsAccel_mps2, 3);

  sensorLog.println("}");
  sensorLog.flush();
}

// ================== 依「第一個 frame 的 t_ms + 日期」切段 MP4 錄影 ==================
void handleMP4Recording(unsigned long frameTimeMs) {
  // 1) 尚未錄影 → 用這個 frame 作為影片的第一幀
  if (!recording) {
    // 優先等 GPS 給時間標記，最多等 30 秒 (以 millis() 計)
    if (!g_timeTagValid && frameTimeMs < 30000) {
      // 還沒 GPS 時間就先不開錄
      return;
    }

    String base;
    if (g_timeTagValid) {
      // 影片檔名：VID_YYYYMMDD_HHMMSS_ms123456.mp4
      base = String("VID_") + String(g_timeTag) + String("_ms") + String(frameTimeMs);
    } else {
      // 沒 GPS 時，退而求其次：VID_ms_123456.mp4
      base = String("VID_ms_") + String(frameTimeMs);
    }

    mp4.setRecordingFileName(base);
    mp4.begin();
    recording = true;
    recordingStartMs = frameTimeMs;

    Serial.print("[MP4] start recording: ");
    Serial.println(base);
  }
  // 2) 已在錄影 → 達到段落時間就結束，下一圈 loop 再開新檔
  else {
    if (frameTimeMs - recordingStartMs >= SEGMENT_MS) {
      mp4.end();
      recording = false;
      Serial.println("[MP4] segment finished, will start new file with new time tag on next frame");
    }
  }
}

// ================== 讀取 MPU6050 ==================
bool readMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  uint8_t txStatus = Wire.endTransmission(false);

  if (txStatus != 0) return false;

  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  delay(1);

  if (Wire.available() < 14) {
    while (Wire.available()) Wire.read();
    return false;
  }

  int16_t ax_raw = (Wire.read() << 8) | Wire.read();
  int16_t ay_raw = (Wire.read() << 8) | Wire.read();
  int16_t az_raw = (Wire.read() << 8) | Wire.read();
  int16_t tempRaw = (Wire.read() << 8) | Wire.read();
  int16_t gx_raw = (Wire.read() << 8) | Wire.read();
  int16_t gy_raw = (Wire.read() << 8) | Wire.read();
  int16_t gz_raw = (Wire.read() << 8) | Wire.read();

  ax_g = ax_raw / ACCEL_SENS;
  ay_g = ay_raw / ACCEL_SENS;
  az_g = az_raw / ACCEL_SENS;

  gx_dps = gx_raw / GYRO_SENS;
  gy_dps = gy_raw / GYRO_SENS;
  gz_dps = gz_raw / GYRO_SENS;

  mpuTempC = (tempRaw / 340.0f) + 36.53f;

  roll_deg  = atan2(ay_g, az_g) * 180.0f / PI;
  pitch_deg = atan2(-ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * 180.0f / PI;

  return true;
}

// ================== 讀取 DHT11 ==================
bool readDHT() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    return false;
  }

  dhtTempC = t;
  dhtHum   = h;
  return true;
}

// ================== GPS 輪詢 ==================
void pollGPS() {
  static String line = "";

  while (Serial1.available()) {
    char c = Serial1.read();
    if (c == '\r') {
      continue;
    } else if (c == '\n') {
      if (line.startsWith("$GPRMC")) {
        parseGPRMC(line);
      } else if (line.startsWith("$GPGGA")) {
        parseGPGGA(line);
      }
      line = "";
    } else {
      if (line.length() < 120) {
        line += c;
      }
    }
  }
}

// ================== GPS 解析 ==================
bool parseLatLon(const String &latField, const String &nsField,
                 const String &lonField, const String &ewField,
                 float &lat, float &lon) {
  if (latField.length() < 4 || lonField.length() < 5) {
    return false;
  }

  float latRaw = latField.toFloat();
  float lonRaw = lonField.toFloat();

  if (latRaw == 0.0f && lonRaw == 0.0f) return false;

  int latDeg = (int)(latRaw / 100);
  float latMin = latRaw - latDeg * 100;
  float latDec = latDeg + latMin / 60.0f;

  int lonDeg = (int)(lonRaw / 100);
  float lonMin = lonRaw - lonDeg * 100;
  float lonDec = lonDeg + lonMin / 60.0f;

  if (nsField == "S") latDec = -latDec;
  if (ewField == "W") lonDec = -lonDec;

  lat = latDec;
  lon = lonDec;
  return true;
}

bool parseGPRMC(const String &sentence) {
  String fields[12];
  int fieldIndex = 0;
  int start = 0;
  int commaPos = -1;

  while (fieldIndex < 12) {
    commaPos = sentence.indexOf(',', start);
    if (commaPos == -1) {
      fields[fieldIndex++] = sentence.substring(start);
      break;
    } else {
      fields[fieldIndex++] = sentence.substring(start, commaPos);
      start = commaPos + 1;
    }
  }

  if (!fields[0].startsWith("$GPRMC")) return false;

  // fields[1] = time hhmmss.sss
  // fields[9] = date ddmmyy
  if (fields[1].length() >= 6 && fields[9].length() >= 6) {
    String t = fields[1]; // hhmmss.sss
    String d = fields[9]; // ddmmyy

    int hh = t.substring(0, 2).toInt();
    int mm = t.substring(2, 4).toInt();
    int ss = t.substring(4, 6).toInt();

    int DD = d.substring(0, 2).toInt();
    int MM = d.substring(2, 4).toInt();
    int YY = d.substring(4, 6).toInt();
    int YYYY = 2000 + YY;   // 簡單假設 20xx 年

    snprintf(g_timeTag, sizeof(g_timeTag),
             "%04d%02d%02d_%02d%02d%02d",
             YYYY, MM, DD, hh, mm, ss);
    g_timeTagValid = true;
  }

  bool hasFix = (fields[2] == "A");

  float lat, lon;
  if (!parseLatLon(fields[3], fields[4], fields[5], fields[6], lat, lon)) {
    gpsHasFix = false;
    return true;
  }

  gpsLatDec = lat;
  gpsLonDec = lon;

  // fields[7] = speed over ground, knots
  float speedKnots = 0.0f;
  if (fields[7].length() > 0) {
    speedKnots = fields[7].toFloat();
  }

  unsigned long nowMs = millis();

  if (hasFix && speedKnots >= 0.0f) {
    float speed_mps = speedKnots * 0.514444f;  // 節 -> m/s

    if (lastSpeedUpdateMs > 0 && !isnan(lastSpeed_mps)) {
      float dt = (nowMs - lastSpeedUpdateMs) / 1000.0f;
      if (dt > 0.05f) {
        gpsAccel_mps2 = (speed_mps - lastSpeed_mps) / dt;
      }
    }

    lastSpeed_mps = speed_mps;
    lastSpeedUpdateMs = nowMs;
    gpsSpeed_mps = speed_mps;
  } else {
    gpsSpeed_mps  = NAN;
    gpsAccel_mps2 = NAN;
  }

  gpsHasFix = hasFix;
  return true;
}

bool parseGPGGA(const String &sentence) {
  String fields[15];
  int fieldIndex = 0;
  int start = 0;
  int commaPos = -1;

  while (fieldIndex < 15) {
    commaPos = sentence.indexOf(',', start);
    if (commaPos == -1) {
      fields[fieldIndex++] = sentence.substring(start);
      break;
    } else {
      fields[fieldIndex++] = sentence.substring(start, commaPos);
      start = commaPos + 1;
    }
  }

  if (!fields[0].startsWith("$GPGGA")) return false;

  int fixQ = fields[6].toInt();  // 0=無定位, 1=GPS fix, 2=DGPS 等
  bool hasFix = (fixQ > 0);

  float lat, lon;
  if (!parseLatLon(fields[2], fields[3], fields[4], fields[5], lat, lon)) {
    gpsHasFix = false;
    return true;
  }

  gpsLatDec = lat;
  gpsLonDec = lon;
  gpsHasFix = hasFix;

  if (!gpsHasFix) {
    gpsSpeed_mps  = NAN;
    gpsAccel_mps2 = NAN;
  }

  return true;
}
