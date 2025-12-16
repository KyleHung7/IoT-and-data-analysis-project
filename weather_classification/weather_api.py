import requests

from datetime import datetime
from typing import Optional, Literal


def _parse_precip(raw: Optional[str]) -> Optional[float]:
    """
    解析降水量欄位（Precipitation）。
    依據中央氣象署資料標準：
    - 單位：毫米 (mm)
    - 特殊代碼：
        X   ：儀器故障
        T   ：雨跡（trace，小於量測門檻）
        -99 ：缺值或資料異常
        -98 ：連續 6 小時無降水
    """
    if raw is None:
        result = {"raw": None, "mm": None, "flag": "missing"}

    raw = str(raw).strip()
    if raw == "":
        result = {"raw": raw, "mm": None, "flag": "missing"}

    # 特殊碼處理
    if raw == "X":
        result = {"raw": raw, "mm": None, "flag": "instrument_error"}
    if raw == "T":
        # trace：有下雨但 < 0.5mm，這裡 mm 給一個代表性的很小數值
        result = {"raw": raw, "mm": 0.0, "flag": "trace"}
    if raw == "-99":
        result = {"raw": raw, "mm": None, "flag": "abnormal"}
    if raw == "-98":
        result = {"raw": raw, "mm": 0.0, "flag": "no_rain_6hr"}

    # 一般數值（毫米）
    try:
        mm_val = float(raw)
        result = {"raw": raw, "mm": mm_val, "flag": "ok"}
    except ValueError:
        # 不預期的字串
        result = {"raw": raw, "mm": None, "flag": "unknown"}
    
    mm_val = result["mm"]
    return mm_val


def get_rainfall(api_key: str, station_id: Optional[str]) -> Optional[float]:
    """
    從中央氣象署 O-A0002-001 取得指定自動雨量站：
    - 過去 1 小時累積降水量

    :param api_key: 氣象資料開放平臺 Authorization 金鑰
    :param station_id: 測站代碼（例如 "C0A560"）
    :return: dict, 例如：
        {
            "stationId": "C0A560",
            "stationName": "某某測站",
            "obsTime": "2025-12-08T12:40:00+08:00",
            "past_1hr":   { "raw": "T",   "mm": 0.0, "flag": "trace" }
        }
    """
    if station_id is None:
        return None

    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"

    params = {"Authorization": api_key}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # O-A0001-001 的官方與教學範例顯示 JSON 結構為 records.Station 陣列，
    # O-A0002-001 使用同一套 Observation 標準，也採用 Station 陣列。 [oai_citation:2‡Medium](https://medium.com/%40gavinkuo123456/%E4%B8%AD%E5%A4%AE%E6%B0%A3%E8%B1%A1%E5%B1%80%E6%89%80%E6%9C%89%E6%B0%A3%E8%B1%A1%E7%AB%99%E5%AF%A6%E6%99%82%E6%BA%AB%E5%BA%A6%E6%95%B8%E6%93%9A%E5%9C%B0%E5%9C%96-b8994235c446?utm_source=chatgpt.com)
    stations = data.get("records", {}).get("Station", [])
    if not stations:
        raise RuntimeError("No Station data found in records")

    # 找到目標測站
    target = None
    for st in stations:
        # 有些資料可能用 'StationId'，有些用 'stationId'，兩種都試
        sid = st.get("StationId") or st.get("stationId")
        if sid == station_id:
            target = st
            break

    if target is None:
        raise ValueError(f"StationId {station_id} not found in O-A0002-001 data")

    rainfall = target.get("RainfallElement", {}) or {}

    # past10_raw = (rainfall.get("Past10Min") or {}).get("Precipitation")
    past1h_raw = (rainfall.get("Past1hr") or {}).get("Precipitation")

    result = _parse_precip(past1h_raw)
    return result


def classify_clear_or_not(weather_text: str | None, precipitation: float | None) -> Literal["clear", "rain", "not_clear"] | None:
    """
    依據 CWA API 回傳的 weather 文字描述與 precipitation 數值，判定天氣狀況為 clear, night, not_clear。
    """
    if weather_text is None:
        if precipitation is None:
            return None
        if precipitation > 0:
            return "not_clear"
        return "clear"
    
    weather_text = weather_text or ""
    if "雪" in weather_text or "冰" in weather_text or "雹" in weather_text or \
        "雨" in weather_text or "雷" in weather_text or "電" in weather_text or \
        "霧" in weather_text or "霾" in weather_text or "靄" in weather_text:
        return "not_clear"

    try:
        precipitation = float(precipitation) if precipitation is not None else 0.0
    except ValueError:
        precipitation = 0.0

    if precipitation > 0:
        return "not_clear"

    return "clear"


def get_weather_condition_api(lat: float, lon: float, api_key: str) -> tuple[Optional[dict], Optional[str]]:
    """
    使用 CWA GraphQL API，依經緯度取得：
    - 觀測時間
    - 觀測站名稱、id、縣市、鄉鎮、座標
    - 原始 weather text / precipitation value
    result = {
        "observed_at": obs_time,
        "station_name": station.get("StationName"),
        "station_id": station_id,
        "county": geo.get("CountyName") or aqi_data.get("county"),
        "town": geo.get("TownName"),
        "latitude": aqi_data.get("latitude"),
        "longitude": aqi_data.get("longitude"),
        "raw_weather": weather_text,
        "past_1hr_precipitation": precipitation,
    }

    :param lat: 緯度
    :param lon: 經度
    :param api_key: CWA 氣象資料開放平臺的 API key
    :return: result, weather condition
    """
    url = "https://opendata.cwa.gov.tw/linked/graphql"

    query = """
    query SimpleWeatherByLocation($lon: Float!, $lat: Float!) {
      aqi(longitude: $lon, latitude: $lat) {
        longitude
        latitude
        sitename
        county
        station {
          StationId
          StationName
          ObsTime {
            DateTime
          }
          GeoInfo {
            CountyName
            TownName
          }
          WeatherElement {
            Weather
            Now {
              Precipitation
            }
          }
        }
      }
    }
    """

    payload = {
        "query": query,
        "variables": {
            "lon": float(lon),
            "lat": float(lat),
        },
    }

    resp = requests.post(
        f"{url}?Authorization={api_key}",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()

    if "errors" in data:
        raise RuntimeError(f"GraphQL error(s): {data['errors']}")

    aqi_data = data.get("data", {}).get("aqi")[0]
    if not aqi_data:
        return None, None
    
    station = aqi_data.get("station") or {}
    station_id = station.get("StationId")

    obs_time = (station.get("ObsTime") or {}).get("DateTime")
    obs_time = datetime.fromisoformat(obs_time) if obs_time else None
    geo = station.get("GeoInfo") or {}
    weather_element = station.get("WeatherElement") or {}
    now = weather_element.get("Now") or {}

    weather_text: str | None = weather_element.get("Weather")
    if weather_text == "-99":
        weather_text = None
    precipitation = get_rainfall(API_KEY, station_id)
    if precipitation == "-99" or precipitation == "X":
        precipitation = None
    elif precipitation == "-98" or precipitation == "T":
        precipitation = 0.0

    condition = classify_clear_or_not(weather_text, precipitation)  # clear, not_clear
    
    if obs_time is not None and condition == "clear" and (obs_time.hour >= 18 or obs_time.hour < 6):
        condition = "night"

    info = {
        "observed_at": obs_time,
        "station_name": station.get("StationName"),
        "station_id": station_id,
        "county": geo.get("CountyName") or aqi_data.get("county"),
        "town": geo.get("TownName"),
        "latitude": aqi_data.get("latitude"),
        "longitude": aqi_data.get("longitude"),
        "raw_weather": weather_text,
        "past_1hr_precipitation": precipitation,
    }
    return info, condition


if __name__ == "__main__":
    API_KEY = "CWA-44F5D81C-9DFF-4792-B054-45ED120B56E6"
    lat = 25.006980997425703
    lon = 121.53587341559226

    info, condition = get_weather_condition_api(lat, lon, API_KEY)
    
    if info is None:
        print("[WARNING] Unable to retrieve weather information.")
    else:
        for k, v in info.items():
            print(f"{k}: {v}")

    if condition is None:
        print("[WARNING] Unable to determine weather condition.")
    else:
        print(f"Weather condition: {condition}")