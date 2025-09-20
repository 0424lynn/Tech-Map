import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import re, os, json, urllib.request

# 【新增】性能相关
from folium.plugins import FastMarkerCluster  # 用于大批量点位聚合（可选）

st.set_page_config(page_title="Tech Map", layout="wide")

# ===================== ① 固定文件夹持久化（读取/保存） =====================
#本地测试
# DATA_DIR_DEFAULT = r"C:\Users\jeffy\chris\tech map"
# 云服务器
DEFAULT_WIN = r"C:\Users\jeffy\chris\tech map"
DATA_DIR_DEFAULT = DEFAULT_WIN if os.path.exists(DEFAULT_WIN) else os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR_DEFAULT, exist_ok=True)
SUPPORT_EXTS = (".csv", ".xlsx", ".xls")

st.sidebar.markdown("### 数据源（固定文件夹）")
data_dir = st.sidebar.text_input("数据文件夹路径", value=DATA_DIR_DEFAULT)
os.makedirs(data_dir, exist_ok=True)

def list_data_files():
    try:
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(SUPPORT_EXTS)]
        files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
        return files
    except Exception:
        return []

def load_df_from_path(path: str) -> pd.DataFrame:
    name = path.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    else:
        return pd.read_excel(path)

def save_uploaded_to_folder(uploaded, folder: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(uploaded.name)
    fname = f"{base}_{ts}{ext}"
    fpath = os.path.join(folder, fname)
    with open(fpath, "wb") as f:
        f.write(uploaded.read())
    return fpath

# session_state 初始化
if "df" not in st.session_state:
    st.session_state.df = None
if "data_meta" not in st.session_state:
    st.session_state.data_meta = {}

# 启动时若无 df，则尝试从文件夹加载最近文件
files = list_data_files()
if st.session_state.df is None and files:
    latest_path = os.path.join(data_dir, files[0])
    try:
        st.session_state.df = load_df_from_path(latest_path)
        st.session_state.data_meta = {
            "filename": files[0],
            "path": latest_path,
            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.info(f"已从文件夹加载最近文件：**{files[0]}**")
    except Exception as e:
        st.error(f"读取 {files[0]} 失败：{e}")

# ======= 侧边栏 · 数据源（折叠面板） =======
with st.sidebar.expander("数据源", expanded=False):
    data_dir = st.text_input(
        "数据文件夹路径",
        value=data_dir if "data_dir" in locals() else DATA_DIR_DEFAULT,
        help="可填本地或共享盘路径，例如 D:\\data 或 \\\\SERVER\\share"
    )
    os.makedirs(data_dir, exist_ok=True)

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(SUPPORT_EXTS)]
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)

    if files:
        pick = st.selectbox("选择已保存的数据文件", files, index=0, key="pick_file")
        if st.button("载入所选文件", key="btn_load_selected"):
            try:
                path = os.path.join(data_dir, pick)
                st.session_state.df = load_df_from_path(path)
                st.session_state.data_meta = {
                    "filename": pick,
                    "path": path,
                    "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success(f"已载入：{pick}")
            except Exception as e:
                st.error(f"载入失败：{e}")
    else:
        st.info("当前文件夹没有任何数据文件（csv/xlsx/xls）。")

    new_file = st.file_uploader("上传新数据（保存进文件夹）", type=['csv', 'xlsx', 'xls'], key="uploader_new")
    if new_file is not None:
        try:
            saved_path = save_uploaded_to_folder(new_file, data_dir)
            st.session_state.df = load_df_from_path(saved_path)
            st.session_state.data_meta = {
                "filename": os.path.basename(saved_path),
                "path": saved_path,
                "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"已上传并载入：{os.path.basename(saved_path)}")
        except Exception as e:
            st.error(f"上传/读取失败：{e}")

    if st.session_state.get("df") is not None:
        meta = st.session_state.get("data_meta", {})
        st.caption("当前数据：")
        st.success(
            f"**{meta.get('filename','(未命名)')}**\n\n"
            f"路径：{meta.get('path','')}\n\n"
            f"载入时间：{meta.get('loaded_at','')}\n\n"
            f"行数：{len(st.session_state.df)}"
        )

# === 统一使用 df（判空） ===
df = st.session_state.get('df')
if df is None:
    st.warning("尚未加载任何数据。请在左侧【数据源】里选择或上传一个文件。")
    st.stop()

# ===================== 【新增】缓存工具（ZIP全量 / 城市郡主表 / 州边界） =====================
@st.cache_data(show_spinner=False)
def load_zip_all_cached():
    import pgeocode
    nomi_local = pgeocode.Nominatim('us')
    z = nomi_local._data[['postal_code','latitude','longitude','state_code','place_name','county_name']].dropna(subset=['latitude','longitude']).copy()
    z['postal_code'] = z['postal_code'].astype(str).str.zfill(5)
    return z

@st.cache_data(show_spinner=False)
def build_city_county_master(zip_all_df: pd.DataFrame):
    cities = (zip_all_df.groupby(['state_code','place_name'], as_index=False)
              .agg(cLat=('latitude','mean'), cLng=('longitude','mean')))
    cities.rename(columns={'state_code':'State','place_name':'City'}, inplace=True)

    counties = (zip_all_df.dropna(subset=['county_name'])
                .groupby(['state_code','county_name'], as_index=False)
                .agg(cLat=('latitude','mean'),
                     cLng=('longitude','mean'),
                     ZIP_count=('postal_code','nunique')))
    counties.rename(columns={'state_code':'State','county_name':'County'}, inplace=True)

    zip_map = (zip_all_df.dropna(subset=['county_name'])
               .groupby(['state_code','county_name'])['postal_code']
               .apply(lambda s: sorted(set(s))).reset_index()
               .rename(columns={'state_code':'State','county_name':'County','postal_code':'ZIPs'}))
    counties = counties.merge(zip_map, on=['State','County'], how='left')
    return cities, counties

@st.cache_data(show_spinner=False)
def load_us_states_geojson_cached(geojson_path):
    if os.path.exists(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as f:
            return json.load(f)
    url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

# ===================== ② 数据清洗 + ZIP/State/City/County 回填 + 官方主表 =====================
# 规范列名
df.columns = [str(c).strip() for c in df.columns]

# 自动识别经纬度列
lat_candidates = ['Latitude', 'Lat', 'latitude', 'lat']
lon_candidates = ['Longitude', 'Lon', 'Lng', 'longitude', 'lon', 'lng']
def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None
lat_col = pick_col(lat_candidates)
lon_col = pick_col(lon_candidates)
if not lat_col or not lon_col:
    st.error(f"未找到经纬度列，请确认列名包含 {lat_candidates} / {lon_candidates}")
    st.stop()

def clean_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(',', '.')
    s = re.sub(r'[^0-9\.\-]', '', s)
    try: return float(s)
    except: return np.nan

df['Latitude']  = df[lat_col].apply(clean_num)
df['Longitude'] = df[lon_col].apply(clean_num)

# Level 容错到 1-6
if 'Level' not in df.columns:
    st.error("缺少必要列：Level")
    st.stop()
def to_level(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+)', str(x))
    if m:
        v = int(m.group(1))
        return v if 1 <= v <= 6 else np.nan   # ← 原来是 <= 5
    return np.nan
df['Level'] = df['Level'].apply(to_level)

# Name/Address 基础校验
for need in ['Name', 'Address']:
    if need not in df.columns:
        st.error(f"缺少必要列：{need}")
        st.stop()

# —— 解析 ZIP 源列（没有就从 Address 中提取） —— #
zip_candidates = ['ZIP', 'Zip', 'zip', 'ZipCode', 'ZIP Code', 'PostalCode', 'Postal Code', 'postcode', 'Postcode', '邮编']
zip_col = next((c for c in zip_candidates if c in df.columns), None)
def extract_zip_from_text(s):
    m = re.search(r'\b(\d{5})(?:-\d{4})?\b', str(s))
    return m.group(1) if m else np.nan

if zip_col is not None:
    df['ZIP'] = df[zip_col]
else:
    df['ZIP'] = df['Address'].apply(extract_zip_from_text)

df['ZIP']  = df['ZIP'].apply(lambda x: extract_zip_from_text(x)).astype('string')
df['ZIP5'] = df['ZIP'].str.zfill(5)

# —— pgeocode：回填点位 State/City/County + 官方“城市/郡”主表 —— #
try:
    import pgeocode
except ImportError:
    st.error("缺少 pgeocode：请先执行  python -m pip install pgeocode")
    st.stop()
nomi = pgeocode.Nominatim('us')

# A) 把 df 的 ZIP5 回填成官方经纬度/州/城市/郡
zip_list = df['ZIP5'].dropna().unique().tolist()
zip_ref = nomi.query_postal_code(zip_list)
zip_ref = zip_ref[['postal_code','latitude','longitude','state_code','place_name','county_name']].dropna(subset=['latitude','longitude'])
zip_ref['postal_code'] = zip_ref['postal_code'].astype(str).str.zfill(5)
df = df.merge(zip_ref, left_on='ZIP5', right_on='postal_code', how='left')

def combine_first_series(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    return a.where(a.notna(), b)

if 'State' not in df.columns:  df['State']  = pd.NA
if 'City'  not in df.columns:  df['City']   = pd.NA
if 'County'not in df.columns:  df['County'] = pd.NA

df['Latitude']  = combine_first_series(df['latitude'],   df['Latitude']).astype(float)
df['Longitude'] = combine_first_series(df['longitude'],  df['Longitude']).astype(float)
df['State']     = combine_first_series(df['state_code'], df['State']).astype('string')
df['City']      = combine_first_series(df['place_name'], df['City']).astype('string')
df['County']    = combine_first_series(df['county_name'],df['County']).astype('string')

df.drop(columns=['postal_code','latitude','longitude','state_code','place_name','county_name'], inplace=True, errors='ignore')

# B) 官方“城市主表 & 郡主表”（来自全美 ZIP） —— 【替换为缓存版】
# zip_all = nomi._data[...]   # ← 原行保留注释
zip_all = load_zip_all_cached()  # 【替换】使用缓存
cities_master, counties_master = build_city_county_master(zip_all)  # 【新增】缓存聚合

# ===================== ③ 侧边栏筛选：郡 / 城市 二选一（默认郡） =====================
st.sidebar.markdown("---")
geo_level = st.sidebar.selectbox("显示范围", ["郡（County）", "城市（City）"], index=0)  # 默认郡

# 等级筛选
levels_present = sorted([int(x) for x in df['Level'].dropna().unique().tolist()]) or [1,2,3,4,5,6]

level_choice = st.sidebar.selectbox('选择等级', ['全部'] + levels_present, index=0)

# 州筛选（从对应主表取）
states_for_level = sorted((counties_master if geo_level.startswith("郡") else cities_master)['State'].unique().tolist())
state_choice = st.sidebar.selectbox('选择州 (State)', ['全部'] + states_for_level)

# 二级下拉：郡 / 城市
if geo_level.startswith("郡"):
    if state_choice != '全部':
        units = sorted(counties_master.loc[counties_master['State']==state_choice, 'County'].unique().tolist())
    else:
        units = sorted(counties_master['County'].unique().tolist())[:5000]
    unit_label = "选择郡 (County)"
else:
    if state_choice != '全部':
        units = sorted(cities_master.loc[cities_master['State']==state_choice, 'City'].unique().tolist())
    else:
        units = sorted(cities_master['City'].unique().tolist())[:5000]
    unit_label = "选择城市 (City)"
unit_choice = st.sidebar.selectbox(unit_label, ['全部'] + units)

# 【新增】性能设置
with st.sidebar.expander("⚡ 性能设置", expanded=False):
    use_cluster = st.checkbox("点位聚合（多点更快）", value=True)
    prefer_canvas = st.checkbox("Canvas 渲染矢量", value=True)
    max_units = st.slider("最多渲染范围数（郡/城市圈）", 200, 5000, 1500, 100)

# 维修工点的基础筛选（用于点位显示）
mask = pd.Series(True, index=df.index)
if level_choice != '全部': mask &= (df['Level'] == level_choice)
if state_choice != '全部': mask &= (df['State'] == state_choice)
if geo_level.startswith("郡") and unit_choice != '全部':
    mask &= (df['County'] == unit_choice)
if (not geo_level.startswith("郡")) and unit_choice != '全部':
    mask &= (df['City'] == unit_choice)
filtered = df.loc[mask].copy()

# —— 优选规则 —— #
st.sidebar.subheader("优选规则")
good_levels = st.sidebar.multiselect("等级筛选", [1,2,3,4,5,6], default=[1,2])

radius_miles = st.sidebar.slider("半径（英里）", 5, 50, 20, 5)
min_good = st.sidebar.number_input("圈内≥ 好维修工数量", 1, 10, 2, 1)
only_show_units = st.sidebar.checkbox("只显示达标范围", value=True)  # 范围=郡或城市
only_show_good_points = st.sidebar.checkbox("只显示“好”的维修工点位", value=False)

# —— 以官方“郡/城市中心”作为圆心，统计圈内维修工 —— #
if geo_level.startswith("郡"):
    base_master = counties_master if state_choice=='全部' else counties_master[counties_master['State']==state_choice]
    name_col = 'County'
    layer_name = "郡圈"
else:
    base_master = cities_master if state_choice=='全部' else cities_master[cities_master['State']==state_choice]
    name_col = 'City'
    layer_name = "城市圈"
if unit_choice != '全部':
    base_master = base_master[base_master[name_col] == unit_choice]
base_master = base_master.copy()

points_all  = df.dropna(subset=['Latitude','Longitude']).copy()
points_good = points_all[points_all['Level'].isin(good_levels)]

# ===== 【优化】BallTree一次建树，避免重复，缺包则回退 =====
R_EARTH_MI = 3958.7613
def counts_balltree(centroids_df, pts_df, radius_mi):
    try:
        from sklearn.neighbors import BallTree
        if pts_df.empty or centroids_df.empty:
            return np.zeros(len(centroids_df), dtype=int)
        P = np.radians(pts_df[['Latitude','Longitude']].to_numpy())
        C = np.radians(centroids_df[['cLat','cLng']].to_numpy())
        tree = BallTree(P, metric='haversine')
        return tree.query_radius(C, r=radius_mi / R_EARTH_MI, count_only=True).astype(int)
    except Exception:
        if pts_df.empty or centroids_df.empty:
            return np.zeros(len(centroids_df), dtype=int)
        latp = np.radians(pts_df['Latitude'].values)
        lonp = np.radians(pts_df['Longitude'].values)
        out = np.zeros(len(centroids_df), dtype=int)
        batch = 1500
        for i in range(0, len(centroids_df), batch):
            clat = np.radians(centroids_df['cLat'].values[i:i+batch])[:, None]
            clng = np.radians(centroids_df['cLng'].values[i:i+batch])[:, None]
            dphi = latp - clat
            dlmb = lonp - clng
            a = np.sin(dphi/2)**2 + np.cos(clat)*np.cos(latp)*np.sin(dlmb/2)**2
            dist = 2*R_EARTH_MI*np.arcsin(np.sqrt(a))
            out[i:i+batch] = (dist <= radius_miles).sum(axis=1)
        return out

# 【新增】优先使用一次建树的方式（若 sklearn 可用）
use_sklearn = True
try:
    from sklearn.neighbors import BallTree
except Exception:
    use_sklearn = False

if use_sklearn:
    P_all = np.radians(points_all[['Latitude','Longitude']].to_numpy()) if len(points_all) else np.empty((0,2))
    P_good = np.radians(points_good[['Latitude','Longitude']].to_numpy()) if len(points_good) else np.empty((0,2))
    C = np.radians(base_master[['cLat','cLng']].to_numpy()) if len(base_master) else np.empty((0,2))
    tree_all = BallTree(P_all, metric='haversine') if len(P_all) else None
    tree_good = BallTree(P_good, metric='haversine') if len(P_good) else None
    r = radius_miles / R_EARTH_MI
    all_cnt  = tree_all.query_radius(C, r=r, count_only=True).astype(int) if tree_all is not None and len(C) else np.zeros(len(base_master), dtype=int)
    good_cnt = tree_good.query_radius(C, r=r, count_only=True).astype(int) if tree_good is not None and len(C) else np.zeros(len(base_master), dtype=int)
    base_master['good_in_radius'] = good_cnt
    base_master['all_in_radius']  = all_cnt
else:
    base_master['good_in_radius'] = counts_balltree(base_master, points_good, radius_miles)
    base_master['all_in_radius']  = counts_balltree(base_master, points_all,  radius_miles)

base_master['meets'] = base_master['good_in_radius'] >= min_good

centroids_to_plot = base_master if not only_show_units else base_master[base_master['meets']]
# 【新增】限制渲染的圈数量，避免全国一次画太多导致卡顿
centroids_to_plot = (centroids_to_plot
                     .sort_values(['meets','good_in_radius','all_in_radius'], ascending=[False, False, False])
                     .head(max_units)
                     .copy())

# 点位集合（可选只显示好工）
points = filtered.dropna(subset=['Latitude','Longitude']).copy()
if only_show_good_points:
    points = points[points['Level'].isin(good_levels)]
n_points = len(points)

# ===================== ④ 地图绘制 =====================
US_STATES_GEO_PATH = os.path.join(data_dir, "us_states.geojson")
def load_us_states_geojson():
    if os.path.exists(US_STATES_GEO_PATH):
        with open(US_STATES_GEO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(US_STATES_GEO_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        st.warning(f"州界数据下载失败：{e}")
        return None

level_color = {
    1:'#2ecc71',  # 绿
    2:'#FFD700',  # 黄
    3:'#FF4D4F',  # 红
    4:'#FFC0CB',  # 粉
    5:'#8A2BE2',  # 紫
    6:'#000000',  # 黑
}

# 初始中心
default_center = [39.5, -98.35]
if n_points:
    center = [points['Latitude'].mean(), points['Longitude'].mean()]
elif not centroids_to_plot.empty:
    center = [centroids_to_plot['cLat'].iloc[0], centroids_to_plot['cLng'].iloc[0]]
else:
    center = default_center

# 【替换】加速：prefer_canvas + 轻底图
m = folium.Map(location=center, zoom_start=6, keyboard=False,
               prefer_canvas=prefer_canvas, tiles="CartoDB positron")  # ← 轻量底图

# 去掉聚焦黑框
m.get_root().header.add_child(folium.Element("""
<style>
.leaflet-container:focus,
.leaflet-container:focus-visible { outline: none !important; }
.leaflet-interactive:focus { outline: none !important; }
</style>
"""))

# —— 州界层：保留边框、无黑色高亮 —— #
# states_geo = load_us_states_geojson()  # ← 原函数
states_geo = load_us_states_geojson_cached(US_STATES_GEO_PATH)  # 【替换】缓存版
selected_bounds = None  # 只放“选中州/全国”的 bounds，给后面缩放用
if states_geo:
    feats = states_geo['features']
    states_fc = {'type': 'FeatureCollection', 'features': feats}

    def style_fn(feat):
        code = feat.get('id') or feat.get('properties', {}).get('state_code')
        is_selected = (state_choice != '全部' and code == state_choice)
        return {
            'fillColor':   '#ffffff',
            'fillOpacity': 0.0,
            'color':       '#2563eb',
            'weight':      3.0 if is_selected else 1.8,
            'dashArray':   None
        }

    gj = folium.GeoJson(
        data=states_fc,
        name="US States",
        style_function=style_fn,
        highlight_function=None,
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'])
    ).add_to(m)

    # 缩放逻辑：选中某州→缩到该州；否则→全国
    if state_choice != '全部':
        target = next((f for f in feats
                       if (f.get('id') or f.get('properties', {}).get('state_code')) == state_choice), None)
        if target:
            def iter_coords(geom):
                # GeoJSON 坐标是 [lng, lat]
                if geom['type'] == 'Polygon':
                    for ring in geom['coordinates']:
                        for lng, lat in ring:
                            yield lat, lng
                elif geom['type'] == 'MultiPolygon':
                    for poly in geom['coordinates']:
                        for ring in poly:
                            for lng, lat in ring:
                                yield lat, lng
            latlngs = list(iter_coords(target['geometry']))
            if latlngs:
                lats, lngs = zip(*latlngs)
                selected_bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]
    else:
        selected_bounds = gj.get_bounds()

# —— 郡/城市圈 —— #
radius_m = radius_miles * 1609.34
unit_fg = folium.FeatureGroup(name=layer_name, show=True).add_to(m)

def preview_zip(zlist, max_show=8):
    if zlist is None or (isinstance(zlist, float) and np.isnan(zlist)): return "0"
    zlist = list(zlist)
    if len(zlist) <= max_show:
        return ",".join(zlist)
    else:
        return ",".join(zlist[:max_show]) + f"... (+{len(zlist)-max_show})"

for _, r in centroids_to_plot.iterrows():
    ring_color = '#1e88e5' if bool(r['meets']) else '#9e9e9e'
    tip = (f"{'County' if geo_level.startswith('郡') else 'City'}: {r.get(name_col)} "
           f"({r.get('State')}) | 好工: {int(r['good_in_radius'])} / 总: {int(r['all_in_radius'])}")
    if geo_level.startswith("郡") and 'ZIP_count' in r and 'ZIPs' in r:
        tip += f" | ZIP数: {int(r['ZIP_count'])} | 示例: {preview_zip(r['ZIPs'])}"
    folium.Circle(
        location=[r['cLat'], r['cLng']],
        radius=radius_m,
        color=ring_color, weight=1.6, fill=True, fill_opacity=0.05,
        tooltip=tip
    ).add_to(unit_fg)

# —— 维修工点位（按等级上色 / 可聚合） —— #
from folium.plugins import MarkerCluster  # 新增导入

workers_fg = folium.FeatureGroup(name="维修工点位", show=True).add_to(m)

if use_cluster and len(points) > 2000:
    # 按等级分组聚合：每个等级一个聚合层，聚合气泡带对应颜色
    clusters = {}
    for lvl, color in level_color.items():
        clusters[lvl] = MarkerCluster(
            name=f"Level {lvl}",
            icon_create_function=f"""
            function(cluster) {{
              var count = cluster.getChildCount();
              return new L.DivIcon({{
                html: '<div style="background:{color};opacity:0.85;border-radius:20px;'
                      +'width:36px;height:36px;display:flex;align-items:center;justify-content:center;'
                      +'color:white;font-weight:600;border:2px solid white;">'+count+'</div>',
                className: 'marker-cluster',
                iconSize: new L.Point(36, 36)
              }});
            }}
            """
        ).add_to(workers_fg)

    # 把点按等级塞到对应聚合层里（保留单点颜色）
    for _, row in points.iterrows():
        lvl = int(row['Level']) if not pd.isna(row['Level']) else None
        color = level_color.get(lvl, '#3388ff')
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup=(f"名称:{row.get('Name','')}<br>"
                   f"等级:{row.get('Level','')}<br>"
                   f"州(State):{row.get('State','')}<br>"
                   f"市/郡:{row.get('City','')}/{row.get('County','')}<br>"
                   f"ZIP:{row.get('ZIP','')}")
        ).add_to(clusters.get(lvl, workers_fg))
else:
    # 原始彩色点
    for _, row in points.iterrows():
        lvl = int(row['Level']) if not pd.isna(row['Level']) else None
        color = level_color.get(lvl, '#3388ff')
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup=(f"名称:{row.get('Name','')}<br>"
                   f"等级:{row.get('Level','')}<br>"
                   f"州(State):{row.get('State','')}<br>"
                   f"市/郡:{row.get('City','')}/{row.get('County','')}<br>"
                   f"ZIP:{row.get('ZIP','')}")
        ).add_to(workers_fg)


# —— 自动缩放到“州界 + 点位 + 范围圈”的联合范围 —— #
def fit_to_union_bounds(map_obj, state_bnds, pts_df, cents_df):
    has_pts = len(pts_df) > 0
    has_c   = len(cents_df) > 0
    if not state_bnds and not has_pts and not has_c:
        return
    b = None
    if state_bnds:
        b = [[state_bnds[0][0], state_bnds[0][1]], [state_bnds[1][0], state_bnds[1][1]]]
    if has_pts:
        pb = [[pts_df['Latitude'].min(), pts_df['Longitude'].min()],
              [pts_df['Latitude'].max(), pts_df['Longitude'].max()]]
        b = pb if b is None else [[min(b[0][0], pb[0][0]), min(b[0][1], pb[0][1])],
                                  [max(b[1][0], pb[1][0]), max(b[1][1], pb[1][1])]]
    if has_c:
        cb = [[cents_df['cLat'].min(), cents_df['cLng'].min()],
              [cents_df['cLat'].max(), cents_df['cLng'].max()]]
        b = cb if b is None else [[min(b[0][0], cb[0][0]), min(b[0][1], cb[0][1])],
                                  [max(b[1][0], cb[1][0]), max(b[1][1], cb[1][1])]]
    if b:
        map_obj.fit_bounds(b)

fit_to_union_bounds(m, selected_bounds, points, centroids_to_plot)

st.caption(
    f"🔹 维修工点位：{n_points:,}  |  🔹 显示范围（{layer_name}）：{len(centroids_to_plot):,} / 全部：{len(base_master):,}  "
    f"(半径 {radius_miles} 英里；好等级={good_levels}；阈值≥{min_good})"
)

# —— 图例 —— #
legend_html = """
<div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999;
            background: white; padding: 10px 12px; border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,.2); font-size: 13px; line-height: 1.6;">
  <b>等级颜色</b><br>
  <span style="display:inline-block;width:12px;height:12px;background:#2ecc71;margin-right:6px;"></span>1 绿色<br>
  <span style="display:inline-block;width:12px;height:12px;background:#FFD700;margin-right:6px;"></span>2 黄色<br>
  <span style="display:inline-block;width:12px;height:12px;background:#FF4D4F;margin-right:6px;"></span>3 红色<br>
  <span style="display:inline-block;width:12px;height:12px;background:#FFC0CB;margin-right:6px;"></span>4 粉色<br>
  <span style="display:inline-block;width:12px;height:12px;background:#8A2BE2;margin-right:6px;"></span>5 紫色<br>
  <span style="display:inline-block;width:12px;height:12px;background:#000000;margin-right:6px;"></span>6 黑色
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=True).add_to(m)

# —— ZIP 列表（郡模式下显示） —— #
if geo_level.startswith("郡"):
    with st.expander("🏷️ 当前州的 郡→ZIP 列表", expanded=False):
        if state_choice == '全部':
            st.info("为避免过多数据，请先选择一个州再查看郡的 ZIP 列表。")
        else:
            show_df = counties_master[counties_master['State']==state_choice][['State','County','ZIP_count']].sort_values('ZIP_count', ascending=False)
            st.dataframe(show_df, use_container_width=True)
            if unit_choice != '全部':
                row = counties_master[(counties_master['State']==state_choice) & (counties_master['County']==unit_choice)]
                if not row.empty:
                    zips = row['ZIPs'].iloc[0]
                    st.write(f"**{unit_choice} County** 的 ZIP（共 {len(zips)} 个）：")
                    st.code(", ".join(zips), language="text")

# —— 渲染 —— #
st_folium(m, width=1400, height=800)
