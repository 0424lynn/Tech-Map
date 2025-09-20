import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import re, os, json, urllib.request
import io  # 用于内存写 Excel

def _build_empty_counties_xlsx(df_to_export: pd.DataFrame) -> io.BytesIO:
    """把空白郡 DataFrame 写入 Excel 并返回内存字节流"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_to_export.to_excel(writer, index=False, sheet_name="EmptyCounties")
    buf.seek(0)
    return buf

# 【新增】性能相关
from folium.plugins import FastMarkerCluster  # 用于大批量点位聚合（可选）

st.set_page_config(page_title="Tech Map", layout="wide")

# 不回传交互事件，完全前端渲染 → 不会出现灰色遮罩
USE_STATIC_MAP = True

SHOW_INLINE_SEARCH = False

st.markdown("""
<style>
/* —— 顶部留白 & 头/脚去掉 —— */
div[data-testid="stDecoration"]{display:none!important;}   /* 顶部彩条 */
header[data-testid="stHeader"]{height:0!important;visibility:hidden!important;}
footer{display:none!important;}
.main .block-container{
  padding-top:.15rem!important;   /* 再次压缩顶部 */
  padding-bottom:.5rem!important;
}

/* —— 隐藏刷新时的灰色遮罩/进度条 —— */
div[role="status"]{display:none!important;}                 /* spinner/status */
div[data-baseweb="progress-bar"]{display:none!important;}   /* 细进度条 */

/* —— 全局字体再小一号（桌面14px，手机13px） —— */
html,body,[class*="css"]{font-size:14px;}
@media (max-width: 640px){
  html,body,[class*="css"]{font-size:13px;}
}

/* —— 输入控件更紧凑 —— */
.stTextInput > div > div > input{height:38px;}
.stTextInput label{margin-bottom:.25rem!important;}

/* —— 下载按钮做成高对比蓝色 —— */
.stDownloadButton button{
  background:#3b82f6!important;color:#fff!important;border:none!important;
  height:42px;border-radius:8px;font-weight:600;
}
.stDownloadButton button:hover{filter:brightness(0.95);}
</style>

<style>
/* —— 手机优化（宽度<=820px） —— */
@media (max-width: 820px){
  /* 主内容与侧边栏内边距更紧凑 */
  .main .block-container{ padding: .35rem .5rem !important; }
  section[data-testid="stSidebar"] > div{ padding-top: .25rem !important; }

  /* 字体整体缩小一点，指标更紧凑 */
  html, body, [class*="css"]{ font-size: 14px !important; }
  [data-testid="stMetricValue"]{ font-size: 1rem !important; }
  [data-testid="stMetricLabel"]{ font-size: .8rem !important; }

  /* 下载按钮更显眼、易点 */
  .stDownloadButton > button, .stButton > button{
    min-height: 38px;
    padding: .5rem .8rem;
    border-radius: 8px;
  }

  /* Folium iframe 自适应手机视口高度（覆盖 st_folium 固定 px） */
  div[data-testid="stIFrame"] iframe{
    width: 100% !important;
    height: 68vh !important;
  }

  /* 新版移动端浏览器用 dvh 更准确 */
  @supports (height: 1dvh){
    div[data-testid="stIFrame"] iframe{ height: 70dvh !important; }
  }

  /* 横向并排控件在手机上改为纵向堆叠 */
  [data-testid="column"]{
    width: 100% !important;
    flex: 0 0 100% !important;
  }
}

/* 仍然隐藏运行时的全屏灰遮罩（兜底） */
.stSpinner, .stSpinnerOverlay, .st-emotion-cache-1erivf3{
  display: none !important;
  opacity: 0 !important;
  pointer-events: none !important;
}

/* 隐藏 Chrome 自动填充按钮（可选） */
input[autocomplete="off"]::-webkit-contacts-auto-fill-button,
input[autocomplete="off"]::-webkit-credentials-auto-fill-button{
  visibility: hidden; display: none !important; pointer-events: none;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* —— 全局基础字号（整体变小） —— */
html, body, .stApp, .main .block-container { font-size: 13px !important; }

/* 侧边栏整体缩小 */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { font-size: 12.5px !important; }

/* 各种表单控件的字 */
.stTextInput, .stNumberInput, .stSelectbox, .stMultiSelect, .stRadio, .stCheckbox, .stSlider, .stDateInput,
.stTextInput *, .stNumberInput *, .stSelectbox *, .stMultiSelect * , .stRadio * , .stCheckbox * , .stSlider * , .stDateInput * {
  font-size: 12.5px !important;
}

/* 按钮 */
.stButton > button { font-size: 12.5px !important; padding: 0.35rem 0.7rem !important; }

/* metric 组件数字稍小 */
[data-testid="stMetricValue"] { font-size: 20px !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }
[data-testid="stMetricLabel"] { font-size: 11.5px !important; }

/* 表格/数据框 */
.stDataFrame, .stDataFrame * { font-size: 12px !important; }

/* 单选/复选横向标签 */
[role="radiogroup"] label, .stCheckbox > label { font-size: 12.5px !important; }

/* Leaflet（Folium）里的字：图层开关、tooltip、popup */
.leaflet-control-layers, .leaflet-container .leaflet-control-attribution,
.leaflet-tooltip, .leaflet-popup-content { font-size: 12px !important; }

/* 让 LayerControl 的标题/选项更紧凑一点 */
.leaflet-control-layers label { line-height: 1.1 !important; }

/* 下载按钮组行内对齐时的字 */
.block-container .row-widget.stButton button { font-size: 12.5px !important; }
</style>
""", unsafe_allow_html=True)


# —— 压缩顶部与分割线间距 —— #
st.markdown("""
<style>
.main .block-container { padding-top: 0.6rem !important; }
section[data-testid="stSidebar"] > div { padding-top: 0.4rem !important; }
hr { margin: 0.3rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 让下载按钮更显眼：蓝底白字 + 轻微阴影 */
.stDownloadButton > button {
  background-color: #2563eb !important;   /* 蓝色 */
  color: #ffffff !important;               /* 白字 */
  border: 1px solid #1d4ed8 !important;
  box-shadow: 0 2px 6px rgba(37,99,235,.25) !important;
  border-radius: 8px !important;
}
.stDownloadButton > button:hover {
  background-color: #1e40af !important;   /* 深一点的蓝（悬停） */
  border-color: #1e3a8a !important;
}
.stDownloadButton > button:active {
  background-color: #1d4ed8 !important;   /* 点击态 */
  transform: translateY(0.5px);
}
</style>
""", unsafe_allow_html=True)


# ===================== ① 固定文件夹持久化（读取/保存） =====================
DEFAULT_WIN = r"C:\Users\jeffy\chris\tech map"
DATA_DIR_DEFAULT = DEFAULT_WIN if os.path.exists(DEFAULT_WIN) else os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR_DEFAULT, exist_ok=True)
SUPPORT_EXTS = (".csv", ".xlsx", ".xls")

# 用 session_state 记住当前数据目录（方便把数据源 UI 放到底部）
if "data_dir_path" not in st.session_state:
    st.session_state.data_dir_path = DATA_DIR_DEFAULT
data_dir = st.session_state.data_dir_path
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
        # st.info(f"已从文件夹加载最近文件：**{files[0]}**")  ← 把这一行注释掉或删除
    except Exception as e:
        st.error(f"读取 {files[0]} 失败：{e}")

# === 统一使用 df（判空） ===
df = st.session_state.get('df')
if df is None:
    st.warning("尚未加载任何数据。请先在页面底部【数据源】处选择或上传一个文件。")
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

# ---------- 自动识别经纬度列（别名 + ZIP/Address 兜底） ----------
alias_map = {}
for c in list(df.columns):
    lc = c.strip().lower()
    if lc in {"lat","latitude","纬度","y","y_coord","ycoordinate","lat_dd","latitudes","lattitude"}:
        alias_map[c] = "Latitude"
    if lc in {"lon","lng","long","longitude","经度","x","x_coord","xcoordinate","lon_dd","longitudes","longtitude"}:
        alias_map[c] = "Longitude"
if alias_map:
    df.rename(columns=alias_map, inplace=True)

lat_candidates = ['Latitude', 'Lat', 'latitude', 'lat']
lon_candidates = ['Longitude', 'Lon', 'Lng', 'longitude', 'lon', 'lng']
def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None
lat_col = pick_col(lat_candidates)
lon_col = pick_col(lon_candidates)

def _extract_zip_from_text(s):
    m = re.search(r'\b(\d{5})(?:-\d{4})?\b', str(s))
    return m.group(1) if m else np.nan

# 若仍未找到，经由 ZIP/Address 回填经纬度
if not lat_col or not lon_col:
    zip_candidates = ['ZIP','Zip','zip','ZipCode','ZIP Code','PostalCode','Postal Code','postcode','Postcode','邮编']
    zip_col0 = next((c for c in zip_candidates if c in df.columns), None)
    if zip_col0 is not None:
        df['ZIP'] = df[zip_col0]
    elif 'Address' in df.columns:
        df['ZIP'] = df['Address'].apply(_extract_zip_from_text)
    else:
        df['ZIP'] = np.nan
    df['ZIP5'] = df['ZIP'].astype('string').str.extract(r'(\d{5})')[0]

    try:
        import pgeocode
        nomi_tmp = pgeocode.Nominatim('us')
        zlist = df['ZIP5'].dropna().unique().tolist()
        if len(zlist) > 0:
            ref = nomi_tmp.query_postal_code(zlist)[['postal_code','latitude','longitude']].dropna()
            ref['postal_code'] = ref['postal_code'].astype(str).str.zfill(5)
            df = df.merge(ref, left_on='ZIP5', right_on='postal_code', how='left')
            def _to_num(x):
                try: return float(str(x).strip().replace(',', '.'))
                except: return np.nan
            df['Latitude']  = pd.to_numeric(df.get('Latitude'), errors='coerce')
            df['Longitude'] = pd.to_numeric(df.get('Longitude'), errors='coerce')
            df['Latitude']  = df['Latitude'].where(df['Latitude'].notna(),  df['latitude'].map(_to_num))
            df['Longitude'] = df['Longitude'].where(df['Longitude'].notna(), df['longitude'].map(_to_num))
            df.drop(columns=['postal_code','latitude','longitude'], inplace=True, errors='ignore')
            lat_col, lon_col = 'Latitude', 'Longitude'
    except Exception as e:
        st.warning(f"基于 ZIP 回填经纬度失败：{e}")

if not lat_col or not lon_col:
    st.error(f"未找到经纬度列，请确认列名包含 {['Latitude','Lat','latitude','lat']} / {['Longitude','Lon','Lng','longitude','lon','lng']}，或提供 Address/ZIP 以便自动推算。当前列：{list(df.columns)}")
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
        return v if 1 <= v <= 6 else np.nan
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

# A) 把 df 的 ZIP5 回填
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

# B) 官方主表
zip_all = load_zip_all_cached()
cities_master, counties_master = build_city_county_master(zip_all)

# ===================== ③ 侧边栏筛选：郡 / 城市 二选一（默认郡） =====================
st.sidebar.markdown("---")
geo_level = st.sidebar.selectbox("显示范围", ["郡（County）", "城市（City）"], index=0)

# 等级筛选
levels_present = sorted([int(x) for x in df['Level'].dropna().unique().tolist()]) or [1,2,3,4,5,6]
level_choice = st.sidebar.selectbox('选择等级', ['全部'] + levels_present, index=0)

# 州、郡/市
states_for_level = sorted((counties_master if geo_level.startswith("郡") else cities_master)['State'].unique().tolist())
state_choice = st.sidebar.selectbox('选择州 (State)', ['全部'] + states_for_level)

if geo_level.startswith("郡"):
    units = sorted(counties_master.loc[counties_master['State']==state_choice, 'County'].unique().tolist()) if state_choice!='全部' \
            else sorted(counties_master['County'].unique().tolist())[:5000]
    unit_label = "选择郡 (County)"
else:
    units = sorted(cities_master.loc[cities_master['State']==state_choice, 'City'].unique().tolist()) if state_choice!='全部' \
            else sorted(cities_master['City'].unique().tolist())[:5000]
    unit_label = "选择城市 (City)"
unit_choice = st.sidebar.selectbox(unit_label, ['全部'] + units)

# 性能设置
with st.sidebar.expander("⚡ 性能设置", expanded=False):
    use_cluster = st.checkbox("点位聚合（多点更快）", value=True)
    prefer_canvas = st.checkbox("Canvas 渲染矢量", value=True)
    max_units = st.slider("最多渲染范围数（郡/城市圈）", 200, 5000, 1500, 100)

# 维修工点位基础筛选
mask = pd.Series(True, index=df.index)
if level_choice != '全部': mask &= (df['Level'] == level_choice)
if state_choice != '全部': mask &= (df['State'] == state_choice)
if geo_level.startswith("郡") and unit_choice != '全部':
    mask &= (df['County'] == unit_choice)
if (not geo_level.startswith("郡")) and unit_choice != '全部':
    mask &= (df['City'] == unit_choice)
filtered = df.loc[mask].copy()

# ==== 顶置搜索（移到更靠上） ====
sc1, sc2 = st.columns([0.5, 0.5])
with sc1:
    q_name = st.text_input(
        "维修工名称",
        key="q_name",
        placeholder="例如：ACME Tech",
        autocomplete="off",      # 不出现下拉联想
        label_visibility="visible",
    )
with sc2:
    q_addr = st.text_input(
        "地址关键词",
        key="q_addr",
        placeholder="城市、州、街道或ZIP",
        autocomplete="off",      # 不出现下拉联想
        label_visibility="visible",
    )

# —— 优选规则 —— #
st.sidebar.subheader("优选规则")
good_levels = st.sidebar.multiselect("等级筛选", [1,2,3,4,5,6], default=[1,2])
radius_miles = st.sidebar.slider("半径（英里）", 5, 50, 20, 5)
min_good = st.sidebar.number_input("圈内≥ 好维修工数量", 1, 10, 2, 1)
only_show_units = st.sidebar.checkbox("只显示达标范围", value=True)
only_show_good_points = st.sidebar.checkbox("只显示“好”的维修工点位", value=False)

# ====== 把“数据源”UI 移到侧边栏最底部 ======
with st.sidebar.expander("📁 数据源（固定文件夹）", expanded=False):
    # 显示并允许修改数据目录
    new_dir = st.text_input(
        "数据文件夹路径",
        value=st.session_state.data_dir_path,
        help="可填本地或共享盘路径，例如 D:\\data 或 \\\\SERVER\\share"
    )
    if new_dir != st.session_state.data_dir_path:
        st.session_state.data_dir_path = new_dir
    os.makedirs(st.session_state.data_dir_path, exist_ok=True)

    files2 = [f for f in os.listdir(st.session_state.data_dir_path) if f.lower().endswith(SUPPORT_EXTS)]
    files2 = sorted(files2, key=lambda f: os.path.getmtime(os.path.join(st.session_state.data_dir_path, f)), reverse=True)

    if files2:
        pick = st.selectbox("选择已保存的数据文件", files2, index=0, key="pick_file_bottom")
        if st.button("载入所选文件", key="btn_load_selected_bottom"):
            try:
                path = os.path.join(st.session_state.data_dir_path, pick)
                st.session_state.df = load_df_from_path(path)
                st.session_state.data_meta = {
                    "filename": pick,
                    "path": path,
                    "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success(f"已载入：{pick}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"载入失败：{e}")
    else:
        st.info("当前文件夹没有任何数据文件（csv/xlsx/xls）。")

    new_file = st.file_uploader("上传新数据（保存进文件夹）", type=['csv', 'xlsx', 'xls'], key="uploader_new_bottom")
    if new_file is not None:
        try:
            saved_path = save_uploaded_to_folder(new_file, st.session_state.data_dir_path)
            st.session_state.df = load_df_from_path(saved_path)
            st.session_state.data_meta = {
                "filename": os.path.basename(saved_path),
                "path": saved_path,
                "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"已上传并载入：{os.path.basename(saved_path)}")
            st.experimental_rerun()
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

# ===================== 进一步筛选：名称/地址搜索（替代 郡→ZIP 列表） =====================
if SHOW_INLINE_SEARCH:
    st.markdown("### 🔍 维修工搜索（名称 / 地址 / 城市 / 郡 / ZIP）")
    name_kw = st.text_input("名称关键词", value="", key="kw_name")
    addr_kw = st.text_input("地址/城市/郡/ZIP 关键词", value="", key="kw_addr")

    # 将搜索与之前的筛选叠加（真正影响地图与“按当前筛选”的统计）
    if name_kw.strip():
        filtered = filtered[filtered['Name'].astype('string').str.contains(name_kw, case=False, na=False)]
    if addr_kw.strip():
        combo = (
            filtered.get('Address', '').astype('string').fillna('') + ' ' +
            filtered.get('City', '').astype('string').fillna('') + ' ' +
            filtered.get('County', '').astype('string').fillna('') + ' ' +
            filtered.get('ZIP', '').astype('string').fillna('') + ' ' +
            filtered.get('State', '').astype('string').fillna('')
        )
        filtered = filtered[combo.str.contains(addr_kw, case=False, na=False)]

    st.caption(f"匹配到 {len(filtered):,} 条。下面预览前 200 条：")
    preview_cols = [c for c in ['Name','Level','State','City','County','ZIP','Address','Latitude','Longitude'] if c in filtered.columns]
    st.dataframe(filtered[preview_cols].head(200), use_container_width=True)

# ===================== 统计圈与点位构建 =====================
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

# BallTree 统计
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

use_sklearn = True
try:
    from sklearn.neighbors import BallTree  # noqa
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
centroids_to_plot = (centroids_to_plot
                     .sort_values(['meets','good_in_radius','all_in_radius'], ascending=[False, False, False])
                     .head(max_units)
                     .copy())


# 作为底图展示的点位集合（受左侧筛选影响）
points = filtered.dropna(subset=['Latitude','Longitude']).copy()
if only_show_good_points:
    points = points[points['Level'].isin(good_levels)]

# 生成“命中集合”，用于放大与加🚩，但不影响 points 的显示
def _contains_safe(s, q):
    return s.astype(str).str.contains(re.escape(q), case=False, na=False)

matched = points.copy()
has_query = False
if q_name:
    has_query = True
    matched = matched[_contains_safe(matched['Name'], q_name)]
if q_addr:
    has_query = True
    addr_mask = (
        _contains_safe(points['Address'], q_addr) |
        _contains_safe(points.get('City',   pd.Series("", index=points.index)),   q_addr) |
        _contains_safe(points.get('County', pd.Series("", index=points.index)),   q_addr) |
        _contains_safe(points.get('ZIP',    pd.Series("", index=points.index)),    q_addr) |
        _contains_safe(points.get('State',  pd.Series("", index=points.index)),  q_addr)
    )
    matched = matched[addr_mask]

# 标记：是否有有效搜索命中
search_active = bool(has_query) and (len(matched) > 0)

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
    1:'#2ecc71', 2:'#FFD700', 3:'#FF4D4F', 4:'#FFC0CB', 5:'#8A2BE2', 6:'#000000'
}

default_center = [39.5, -98.35]
if n_points:
    center = [points['Latitude'].mean(), points['Longitude'].mean()]
elif not centroids_to_plot.empty:
    center = [centroids_to_plot['cLat'].iloc[0], centroids_to_plot['cLng'].iloc[0]]
else:
    center = default_center

m = folium.Map(location=[37.8, -96.0], zoom_start=4, keyboard=False,
               prefer_canvas=prefer_canvas, tiles="CartoDB positron")

m.get_root().header.add_child(folium.Element("""
<style>
.leaflet-container:focus,
.leaflet-container:focus-visible { outline: none !important; }
.leaflet-interactive:focus { outline: none !important; }
</style>
"""))

states_geo = load_us_states_geojson_cached(US_STATES_GEO_PATH)
selected_bounds = None
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
        data=states_fc, name="US States",
        style_function=style_fn, highlight_function=None,
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'])
    ).add_to(m)

    if state_choice != '全部':
        target = next((f for f in feats
                       if (f.get('id') or f.get('properties', {}).get('state_code')) == state_choice), None)
        if target:
            def iter_coords(geom):
                if geom['type'] == 'Polygon':
                    for ring in geom['coordinates']:
                        for lng, lat in ring: yield lat, lng
                elif geom['type'] == 'MultiPolygon':
                    for poly in geom['coordinates']:
                        for ring in poly:
                            for lng, lat in ring: yield lat, lng
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
        location=[r['cLat'], r['cLng']], radius=radius_m,
        color=ring_color, weight=1.6, fill=True, fill_opacity=0.05,
        tooltip=tip
    ).add_to(unit_fg)

# —— 维修工点位 —— #
from folium.plugins import MarkerCluster
workers_fg = folium.FeatureGroup(name="维修工点位", show=True).add_to(m)

if use_cluster and len(points) > 2000:
    clusters = {}
    for lvl, color in level_color.items():
        clusters[lvl] = MarkerCluster(
            name=f"Level {lvl}",
            icon_create_function=f"""
            function(cluster) {{
              var count = cluster.getChildCount();
              return new L.DivIcon({{
                html: '<div style="background:{color};opacity:0.85;border-radius:20px;width:36px;height:36px;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;border:2px solid white;">'+count+'</div>',
                className: 'marker-cluster', iconSize: new L.Point(36, 36)
              }});
            }}
            """
        ).add_to(workers_fg)

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

      # —— 大号红旗（锚点对齐版）——
def big_flag_icon(size_px: int = 42, anchor_y_factor: float = 0.92) -> folium.DivIcon:
    anchor_y = int(size_px * anchor_y_factor)   # 0.88~0.96 之间微调
    return folium.DivIcon(
        html=f'''
        <div style="filter: drop-shadow(0 0 1px #fff) drop-shadow(0 0 6px rgba(0,0,0,.35));">
          <span style="font-size:{size_px}px; line-height:1;">🚩</span>
        </div>
        ''',
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, anchor_y)  # 底部中点 ≈ 旗尖
    )

# 在你完成所有点位（workers_fg / clusters）之后，加这一段：
if 'search_active' in locals() and search_active and 'matched' in locals() and len(matched) > 0:
    for _, r in matched.iterrows():
        folium.Marker(
            location=[float(r['Latitude']), float(r['Longitude'])],
            icon=big_flag_icon(size_px=42),         # ← 想更大就调这个值，如 48/56
            tooltip=f"🔎 命中：{r.get('Name','')}",
            popup=(f"<b>名称：</b>{r.get('Name','')}<br>"
                   f"<b>地址：</b>{r.get('Address','')}<br>"
                   f"<b>等级：</b>{r.get('Level','')}"),
            z_index_offset=10000                    # 永远盖在最上层
        ).add_to(m)


# —— 自动缩放 —— #
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

# —— 智能缩放 —— #
# 固定全美（CONUS）外接矩形：下边界 24.5°N，上边界 49.5°N；西界 -125°E，东界 -66.9°E
CONUS_BOUNDS = [[24.5, -125.0], [49.5, -66.9]]

def fit_initial_or_search(map_obj, nat_bnds, state_bnds, matched_df, search_active):
    # 1) 若有搜索命中：缩放到命中的维修工（多条则包络，单条给一点 buffer）
    if search_active and matched_df is not None and len(matched_df) > 0:
        if len(matched_df) == 1:
            lat = float(matched_df['Latitude'].iloc[0])
            lng = float(matched_df['Longitude'].iloc[0])
            pad = 0.35
            b = [[lat - pad, lng - pad], [lat + pad, lng + pad]]
        else:
            b = [[matched_df['Latitude'].min(),  matched_df['Longitude'].min()],
                 [matched_df['Latitude'].max(),  matched_df['Longitude'].max()]]
        map_obj.fit_bounds(b)
        return

    # 2) 若选中某州：按该州边界缩放
    if (state_choice != '全部') and state_bnds:
        map_obj.fit_bounds(state_bnds)
        return

    # 3) 默认：全美（CONUS）
    map_obj.fit_bounds(nat_bnds)

# 这里 matched / search_active 来自你前面“搜索栏”那段（命中集合）
fit_initial_or_search(
    m,
    CONUS_BOUNDS,
    selected_bounds,                         # 有选州时的州边界
    matched if 'matched' in locals() else None,
    search_active if 'search_active' in locals() else False
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

# === 导出：当前筛选下“没有任何维修工”的郡/城市（按半径+阈值） ===
st.markdown("---")

# 按当前地理层级决定标签
unit_key = 'County' if geo_level.startswith("郡") else 'City'
label_total   = "统计的郡数" if unit_key == 'County' else "统计的城市数"
label_yes     = "有维修工的郡" if unit_key == 'County' else "有维修工的城市"
label_no      = "没有维修工的郡" if unit_key == 'County' else "没有维修工的城市"

# 用你前面已经构建好的 base_master（已按州/郡/市筛选，且包含 good_in_radius / meets）
cm_units = base_master.copy()

# 指标：随 radius_miles / min_good / good_levels 实时变化
total_units  = len(cm_units)
covered_units = int(cm_units['meets'].sum())          # 达标（圈内≥min_good 的“好工”）
empty_units   = total_units - covered_units           # 不达标
empty_rate    = (empty_units / total_units) if total_units else 0

# 顶部一行：三枚指标 + 右侧下载
left, mid, right, dl = st.columns([0.9, 0.9, 0.9, 1.1])

with left:
    st.metric(label_total, f"{total_units:,}")
with mid:
    st.metric(label_yes, f"{covered_units:,}")
with right:
    st.metric(label_no, f"{empty_units:,}", f"{empty_rate:.1%} 空白率")

# 导出“不达标”的范围列表（即 meets == False）
gaps = cm_units[~cm_units['meets']].copy()
# 为了兼容你之前的列名，这里把圈内“好工”数写到 workers_count
gaps['workers_count'] = gaps['good_in_radius'].astype(int)


 # —— 指标后的导出设置 —— 
unit_key = 'County' if geo_level.startswith("郡") else 'City'

if unit_key == 'County':
    outcols   = ["State", "County", "ZIP_count", "ZIPs", "cLat", "cLng", "workers_count"]
    sort_cols = ["State", "County"]
else:
    outcols   = ["State", "City", "cLat", "cLng", "workers_count"]
    sort_cols = ["State", "City"]

# 只保留确实存在的列，避免 KeyError
outcols_present = [c for c in outcols if c in gaps.columns]
gaps_sorted = gaps[outcols_present].sort_values(sort_cols).copy()

# 县模式下把 ZIPs 列表展开为字符串（逗号分隔）
if "ZIPs" in gaps_sorted.columns:
    def _z_to_str(z):
        if isinstance(z, (list, tuple, set, np.ndarray)):
            return ", ".join(sorted(map(str, z)))
        return "" if pd.isna(z) else str(z)
    gaps_sorted["ZIPs"] = gaps_sorted["ZIPs"].apply(_z_to_str)

# 生成下载文件
tag    = f"r{int(radius_miles)}_min{int(min_good)}"
prefix = "counties" if unit_key == "County" else "cities"
fname  = f"{prefix}_not_meeting_threshold_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
excel_bytes = _build_empty_counties_xlsx(gaps_sorted)


with dl:
    clicked = st.download_button(
        "下载",
        data=excel_bytes,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_top"
    )
if clicked:
        try:
            save_path = os.path.join(data_dir, fname)
            with open(save_path, "wb") as f:
                f.write(excel_bytes.getbuffer())
            st.toast(f"已保存到：{save_path}")
        except Exception as e:
            st.warning(f"无法保存到本地文件夹：{e}")

   
# —— 渲染 —— #
if globals().get('USE_STATIC_MAP'):
    from streamlit.components.v1 import html
    html(m.get_root().render(), height=800)  # 无灰屏、放大/缩小不触发 rerun
else:
    map_height = st.session_state.get("map_height", 800)  # 桌面基准高度；手机CSS会调
    st_folium(m, use_container_width=True, height=map_height)

