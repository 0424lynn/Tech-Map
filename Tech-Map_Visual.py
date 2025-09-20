import pandas as pd
import folium

# 读取筛选结果
df = pd.read_excel('Tech-Map_Filtered.xlsx')

# 以第一个点为地图中心
center_lat = df['Latitude'].mean()
center_lng = df['Longitude'].mean()
map_ = folium.Map(location=[center_lat, center_lng], zoom_start=10)

# 添加维修工点
for _, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Name']} 等级:{row['Level']} 城市:{row['city_center']}",
        icon=folium.Icon(color='blue', icon='wrench', prefix='fa')
    ).add_to(map_)

# 保存为HTML
map_.save('Tech-Map_Visual.html')
print('地图已生成：Tech-Map_Visual.html')