import requests

def classify_sun_or_rain(weather_text: str | None, precipitation: float | None) -> str | None:
    """
    依據 CWA API 回傳的 weather 文字描述與 precipitation 數值，判定天氣狀況為 clear, rain, snow, fog。
    """
    if weather_text is None:
        return None
    weather_text = weather_text or ""
    if "雪" in weather_text or "冰" in weather_text or "雹" in weather_text:
        return "snow"
    if "雨" in weather_text or "雷" in weather_text or "電" in weather_text:
        return "rain"
    if "霧" in weather_text or "霾" in weather_text or "靄" in weather_text:
        return "fog"

    try:
        precip = float(precipitation) if precipitation is not None else 0.0
    except ValueError:
        precip = 0.0

    if precip > 0:
        return "rain"

    return "clear"


def get_weather(lat: float, lon: float, api_key: str) -> dict | None:
    """
    使用 CWA GraphQL API，依經緯度取得：
    - 判定後的天氣狀況（晴/雨）
    - 觀測時間
    - 觀測站名稱
    - 縣市、鄉鎮
    - 座標
    - 原始 weather/precipitation

    :param lat: 緯度
    :param lon: 經度
    :param api_key: CWA 氣象資料開放平臺的 Authorization 金鑰
    :return: 上述欄位組成的 dict；若查不到資料則回傳 None
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
        return None
    
    station = aqi_data.get("station") or {}
    obs_time = (station.get("ObsTime") or {}).get("DateTime")
    geo = station.get("GeoInfo") or {}
    weather_element = station.get("WeatherElement") or {}
    now = weather_element.get("Now") or {}

    weather_text: str | None = weather_element.get("Weather")
    precipitation = now.get("Precipitation")

    condition = classify_sun_or_rain(weather_text, precipitation)

    result = {
        "condition": condition,  # clear, rain, snow, fog
        "observed_at": obs_time,
        "station_name": station.get("StationName"),
        "county": geo.get("CountyName") or aqi_data.get("county"),
        "town": geo.get("TownName"),
        "longitude": aqi_data.get("longitude"),
        "latitude": aqi_data.get("latitude"),
        "raw_weather": weather_text,
        "raw_precipitation": precipitation,
    }
    return result


if __name__ == "__main__":
    API_KEY = "CWA-44F5D81C-9DFF-4792-B054-45ED120B56E6"
    lat = 25.006980997425703
    lon = 121.53587341559226

    info = get_weather(lat, lon, API_KEY)
    print(info)