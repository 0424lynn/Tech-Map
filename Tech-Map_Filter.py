import pandas as pd
import numpy as np
from geopy.distance import geodesic

# 读取数据
file_path = 'Tech-Map Data with LatLng.csv'
df = pd.read_csv(file_path)

# 提取城市字段（假设地址格式为"城市..."）
df['City'] = df['Address'].str.extract(r'(\w+)')

# 支持等级筛选和名称模糊搜索
level_selected = st.sidebar.multiselect('选择等级', sorted(df['Level'].unique()))
name_keyword = st.sidebar.text_input('名称模糊搜索')
if level_selected:
    df = df[df['Level'].isin(level_selected)]
if name_keyword:
    df = df[df['Name'].str.contains(name_keyword, case=False)]

# 按城市分组，计算中心点
city_groups = df.groupby('City')
writer = pd.ExcelWriter('Tech-Map_Filtered.xlsx')
for city, group in city_groups:
    center = (group['Latitude'].mean(), group['Longitude'].mean())
    # 计算距离
    group['Distance'] = group.apply(lambda x: geodesic(center, (x['Latitude'], x['Longitude'])).miles, axis=1)
    # 距离筛选（可选25/50/75英里）
    for radius in [25, 50, 75]:
        filtered = group[group['Distance'] <= radius]
        sheet_name = f"{city}_{radius}mi"
        filtered.to_excel(writer, sheet_name=sheet_name, index=False)
writer.save()