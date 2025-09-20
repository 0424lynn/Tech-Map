import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut

# 读取CSV
input_file = 'Tech-Map Data.csv'
df = pd.read_csv(input_file)

# 初始化Nominatim
geolocator = Nominatim(user_agent="tech_map_geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=10, error_wait_seconds=10)

def get_latitude(address):
    for i in range(5):  # 最多重试5次
        try:
            location = geocode(address)
            if location:
                return location.latitude
        except GeocoderTimedOut:
            time.sleep(3)  # 超时后等待3秒再试
    return None

def get_longitude(address):
    for i in range(5):
        try:
            location = geocode(address)
            if location:
                return location.longitude
        except GeocoderTimedOut:
            time.sleep(3)
    return None

# 新增经纬度列
df['Latitude'] = df['Address'].apply(get_latitude)
df['Longitude'] = df['Address'].apply(get_longitude)

# 保存新文件
df.to_csv('Tech-Map Data with LatLng.csv', index=False)
print('已完成地理编码，结果保存在Tech-Map Data with LatLng.csv')