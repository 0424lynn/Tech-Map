# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import re, os, json, urllib.request
import io
import requests, time
from folium.plugins import MarkerCluster

# ======================
# 基础设置
# ======================
st.set_page_config(page_title="Tech Map", layout="wide")
USE_STATIC_MAP = True  # Folium 用原生 HTML 渲染，更快
# ---- 兼容 rerun（新：st.rerun；旧：st.experimental_rerun）----
def _safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ---------- 正则 ----------
HVAC_PAT_STR = (
    r"(hvac|air\s*conditioning|\bac\b|a/?c|heating|heat\s*pump|furnace|boiler|"
    r"refrigeration|cooling|ventilation|duct|chiller|mini\s*split|split\s*system|"
    r"thermo\s*stat|compressor|refrigerant|制冷|制热|暖通|空调|冷冻|冷库|冷气|冷暖|暖气|通风)"
)
BLACK_PAT_STR = (
    r"(?:at[&\s]*t|verizon|t[-\s]*mobile|cricket|boost\s*mobile|"
    r"metro(?:\s*pcs|\s*by\s*t[-\s]*mobile)?|xfinity|comcast|spectrum|sprint|"
    r"u\.?\s*s\.?\s*cellular|cell(?:ular)?|mobile|phone|iphone|android|"
    r"computer|pc|laptop|electronics|tv|best\s*buy|geek\s*squad|apple\s*store|"
    r"samsung|gamestop|"
    r"auto|car|truck|dealer(?:ship)?|motor\s*sports|body\s*shop|collision|"
    r"brake|tire|alignment|transmission|oil\s*change|smog|muffler|lube|"
    r"towing|tow|dismantler|parts?|quick\s*lube|"
    r"pep\s*boys|firestone|goodyear|jiffy\s*lube|valvoline|mobil\s*1|"
    r"honda|toyota|chevrolet|ford"
    r")"
)
HVAC_RE  = re.compile(HVAC_PAT_STR,  re.IGNORECASE)
BLACK_RE = re.compile(BLACK_PAT_STR, re.IGNORECASE)

# ---------- 图标 ----------
def blue_wrench_icon(size_px: int = 24):
    return folium.DivIcon(
        html=f"""
        <div style="filter: drop-shadow(0 0 2px rgba(0,0,0,.35));">
          <span style="font-size:{size_px}px; line-height:1;">🔧</span>
        </div>
        """,
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, int(size_px * 0.92)),
    )

def big_flag_icon(size_px: int = 42, anchor_y_factor: float = 0.92):
    anchor_y = int(size_px * anchor_y_factor)
    return folium.DivIcon(
        html=f'''
        <div style="filter: drop-shadow(0 0 1px #fff) drop-shadow(0 0 6px rgba(0,0,0,.35));">
          <span style="font-size:{size_px}px; line-height:1;">🚩</span>
        </div>
        ''',
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, anchor_y)
    )

def customer_pin_icon(size_px: int = 42):
    # 与🔧视觉一致，默认 42px
    return folium.DivIcon(
        html=f"""
        <div style="filter: drop-shadow(0 0 2px rgba(0,0,0,.35));">
          <span style="font-size:{size_px}px; line-height:1;">📍</span>
        </div>
        """,
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, int(size_px * 0.92)),
    )

# ---------- 读取 Google Key ----------
def _read_api_key():
    key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if key:
        return key
    candidate_paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
        os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(p) for p in candidate_paths):
        try:
            return st.secrets.get("GOOGLE_PLACES_API_KEY", None)
        except Exception:
            pass
    return None

GOOGLE_PLACES_KEY = _read_api_key()
def _mask_key(k: str, keep=6):
    if not k: return ""
    k = str(k).strip()
    return k[:keep] + "…" + str(len(k))

# ======================
# 全局样式（紧凑/边框）
# ======================
st.markdown("""
<style>
/* remove chrome */
div[data-testid="stDecoration"]{display:none!important;}
header[data-testid="stHeader"]{height:0!important;visibility:hidden!important;}
footer{display:none!important;}
:root, .stApp { --top-toolbar-height:0px !important; }

/* layout tighten */
.stAppViewContainer{ padding-top:0!important; }
.main .block-container{ padding-top:.1rem!important; margin-top:0!important; }
.main .block-container > div{ margin-top:.3rem!important; }

/* row spacing tighter under search row */
div[data-testid="stHorizontalBlock"]{ margin-bottom:.1rem!important; }

/* fonts */
html, body, .stApp, .main .block-container { font-size: 13px !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { font-size: 12.5px !important; }

/* metrics smaller */
[data-testid="stMetricValue"] { font-size: 18px !important; line-height:1.1!important; font-weight:700!important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; line-height:1.1!important; }
[data-testid="stMetricDelta"] { font-size: 10px !important; line-height:1.1!important; }
div[data-testid="stMetric"] > div { padding-top:2px!important; padding-bottom:2px!important; margin:0!important; }

/* map iframe closer to top */
div[data-testid="stIFrame"]{ margin-top: .1rem!important; }

/* buttons */
.stDownloadButton > button{
  background:#2563eb!important; color:#fff!important;
  border:1px solid #1d4ed8!important; border-radius:8px!important;
  box-shadow:0 2px 6px rgba(37,99,235,.25)!important;
}
.stDownloadButton > button:hover { background-color:#1e40af!important; border-color:#1e3a8a!important; }
.stDownloadButton > button:active { background-color:#1d4ed8!important; transform: translateY(0.5px); }

/* subtle borders for a less-gray feel */
.main .block-container > div{
  border:1px solid rgba(0,0,0,.06);
  border-radius:10px;
  padding:.6rem .7rem;
  background:#fff;
  box-shadow:0 1px 4px rgba(0,0,0,.04);
}
</style>
""", unsafe_allow_html=True)

# ======================
# 数据目录（本机优先，云端用仓库 data/；也支持环境变量覆盖）
# ======================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_DIR = r"C:\Users\jeffy\chris\tech map"  # 本机跑时可用
DATA_DIR_ENV = os.getenv("TECH_MAP_DATA_DIR")

DATA_DIR_DEFAULT = DATA_DIR_ENV or (LOCAL_DIR if os.path.exists(LOCAL_DIR) else os.path.join(APP_DIR, "data"))
os.makedirs(DATA_DIR_DEFAULT, exist_ok=True)

SUPPORT_EXTS = (".csv", ".xlsx", ".xls")

if "data_dir_path" not in st.session_state:
    st.session_state.data_dir_path = DATA_DIR_DEFAULT

data_dir = st.session_state.data_dir_path
os.makedirs(data_dir, exist_ok=True)


def _list_files():
    try:
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(SUPPORT_EXTS)]
        return sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    except Exception:
        return []

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    return pd.read_excel(path)

def _save_uploaded(uploaded, folder: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(uploaded.name)
    fpath = os.path.join(folder, f"{base}_{ts}{ext}")
    with open(fpath, "wb") as f:
        f.write(uploaded.read())
    return fpath

# ======================
# Session 初始化
# ======================
if "df" not in st.session_state:
    st.session_state.df = None
if "data_meta" not in st.session_state:
    st.session_state.data_meta = {}

# 启动时自动加载最近文件
_files = _list_files()
if st.session_state.df is None and _files:
    latest_path = os.path.join(data_dir, _files[0])
    try:
        st.session_state.df = _load_df(latest_path)
        st.session_state.data_meta = {
            "filename": _files[0],
            "path": latest_path,
            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error(f"读取 {_files[0]} 失败：{e}")

df = st.session_state.get("df")
# 侧边栏底部：数据源/Key
st.markdown("---")
with st.expander("📁 数据源（固定文件夹）", expanded=False):
        new_dir = st.text_input("数据文件夹路径", value=st.session_state.data_dir_path)
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
                    st.session_state.df = _load_df(path)
                    st.session_state.data_meta = {"filename": pick, "path": path, "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    st.success(f"已载入：{pick}")
                    _safe_rerun()
                except Exception as e:
                    st.error(f"载入失败：{e}")
        else:
            st.info("当前文件夹没有任何数据文件（csv/xlsx/xls）。")

        new_file = st.file_uploader("上传新数据（保存进文件夹）", type=['csv', 'xlsx', 'xls'], key="uploader_new_bottom")
        if new_file is not None:
            try:
                saved_path = _save_uploaded(new_file, st.session_state.data_dir_path)
                st.session_state.df = _load_df(saved_path)
                st.session_state.data_meta = {"filename": os.path.basename(saved_path), "path": saved_path, "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                st.success(f"已上传并载入：{os.path.basename(saved_path)}")
                _safe_rerun()
            except Exception as e:
                st.error(f"上传/读取失败：{e}")

        if st.session_state.get("df") is not None:
            meta = st.session_state.get("data_meta", {})
            st.success(
                f"**{meta.get('filename','(未命名)')}**\n\n"
                f"路径：{meta.get('path','')}\n\n"
                f"载入时间：{meta.get('loaded_at','')}\n\n"
                f"行数：{len(st.session_state.df)}"
            )

if df is None:
    st.warning("尚未加载任何数据。请到侧边栏最底部【📁 数据源（固定文件夹）】选择或上传文件。")
    st.stop()

# ======================
# 缓存 / 参考表
# ======================
@st.cache_data(show_spinner=False)
def load_zip_all_cached():
    import pgeocode
    nomi_local = pgeocode.Nominatim("us")
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
               .apply(lambda s: sorted(set(s))).reset_index()\
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

# ---- 更稳的地址解析（坐标/ZIP/Google/OSM 多重兜底）----
ZIP_RE = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
LATLNG_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$')

# 50 州缩写（用于辅助判断 5 位数字是否真的是 ZIP）
_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
    "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
    "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR"
}

def _smart_zip_from_text(s: str):
    """优先匹配 '州缩写 + ZIP'，否则取最后一个5位数为 ZIP。对 pd.NA 安全。"""
    # 先把各种空值都拦住：None / NaN / pd.NA / 空串
    try:
        import pandas as pd  # 已经有就不影响
        if s is None or pd.isna(s):
            return None
    except Exception:
        if s is None:
            return None

    s = str(s).strip()
    if not s:
        return None

    # 先找形如 "MI 48219" / "MI, 48219"
    m = re.search(r'\b([A-Z]{2})\s*,?\s*(\d{5})(?:-\d{4})?\b', s)
    if m and m.group(1).upper() in _US_STATES:
        return m.group(2)

    # 否则取“最后一个”5位数
    ms = list(re.finditer(r'\b(\d{5})(?:-\d{4})?\b', s))
    return ms[-1].group(1) if ms else None


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address(addr: str, key: str | None):
    addr = (addr or "").strip()
    if not addr:
        return None

    # ① 坐标
    m = LATLNG_RE.match(addr)
    if m:
        lat = float(m.group(1)); lng = float(m.group(2))
        return {"lat": lat, "lng": lng, "formatted": f"{lat:.6f}, {lng:.6f}", "source": "coord"}

       # ② ZIP（使用更智能的提取，避免把门牌号当 ZIP）
    zip5 = _smart_zip_from_text(addr)
    if zip5:
        try:
            import pgeocode
            nomi = pgeocode.Nominatim("us")
            info = nomi.query_postal_code(zip5)
            if pd.notna(info.latitude) and pd.notna(info.longitude):
                return {
                    "lat": float(info.latitude), "lng": float(info.longitude),
                    "formatted": f"ZIP {zip5}",
                    "source": "zip"
                }
        except Exception:
            pass


    # ③ Google
    if key:
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": addr, "key": key, "language": "en"},
                timeout=15
            )
            j = r.json()
            if j.get("status") == "OK" and j.get("results"):
                g = j["results"][0]; loc = g["geometry"]["location"]
                return {
                    "lat": float(loc["lat"]), "lng": float(loc["lng"]),
                    "formatted": g.get("formatted_address", addr),
                    "source": "google"
                }
            else:
                raise RuntimeError(f"Google: {j.get('status')} {j.get('error_message','')}")
        except Exception:
            pass

    # ④ OSM / ⑤ maps.co
    headers = {"User-Agent": "tech-map/1.0 (contact: support@example.com)"}
    osm_params = {"q": addr, "format": "json", "limit": 1, "addressdetails": 1, "countrycodes": "us", "accept-language": "en"}
    for url, src in [
        ("https://nominatim.openstreetmap.org/search", "osm"),
        ("https://geocode.maps.co/search", "mapsco"),
    ]:
        try:
            r2 = requests.get(url, params=osm_params, headers=headers, timeout=20)
            arr = r2.json()
            if isinstance(arr, list) and arr:
                it = arr[0]
                return {
                    "lat": float(it["lat"]), "lng": float(it["lon"]),
                    "formatted": it.get("display_name", addr),
                    "source": src
                }
        except Exception:
            pass
    return None

# ---- 客户地址：回车触发定位（无按钮）----
def _on_cust_addr_change():
    addr = st.session_state.get("cust_quick_addr", "").strip()
    if not addr:
        st.session_state.pop("cust_quick_pin", None)
        return
    gq = geocode_address(addr, GOOGLE_PLACES_KEY)
    if not gq:
        st.session_state["_last_cust_geocode_msg"] = "地址解析失败：请换个写法或补充城市/州，或直接输入坐标，如 34.0522,-118.2437。"
        st.session_state.pop("cust_quick_pin", None)
        return
    st.session_state['cust_quick_pin'] = {"lat": gq["lat"], "lng": gq["lng"], "formatted": gq.get("formatted", addr), "source": gq.get("source","")}
    pad = 0.2
    st.session_state["_zoom_bounds"] = [[gq["lat"]-pad, gq["lng"]-pad],[gq["lat"]+pad, gq["lng"]+pad]]
    src = gq.get("source","geocoder")
    st.session_state["_last_cust_geocode_msg"] = f"已定位到：{gq.get('formatted', addr)}（来源：{src}）"

# ======================
# 数据清洗/回填
# ======================
df.columns = [str(c).strip() for c in df.columns]
alias_map = {}
for c in list(df.columns):
    lc = c.lower()
    if lc in {"lat","latitude","纬度","y","y_coord","ycoordinate","lat_dd","latitudes","lattitude"}:
        alias_map[c] = "Latitude"
    if lc in {"lon","lng","long","longitude","经度","x","x_coord","xcoordinate","lon_dd","longitudes","longtitude"}:
        alias_map[c] = "Longitude"
if alias_map:
    df.rename(columns=alias_map, inplace=True)

def _pick_col(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None
lat_col = _pick_col(['Latitude', 'Lat', 'latitude', 'lat'])
lon_col = _pick_col(['Longitude', 'Lon', 'Lng', 'longitude', 'lon', 'lng'])

def _extract_zip_from_text(s):
    z = _smart_zip_from_text(s)
    # 统一返回为缺失值（string dtype 友好）
    try:
        import pandas as pd
        if z is None or pd.isna(z) or str(z).strip() == "":
            return pd.NA
    except Exception:
        if z is None or str(z).strip() == "":
            return np.nan
    return z


# 经纬度回填 ZIP

# ZIP 回填经纬度（若没提供经纬度）
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
        if zlist:
            ref = nomi_tmp.query_postal_code(zlist)[['postal_code','latitude','longitude']].dropna()
            ref['postal_code'] = ref['postal_code'].astype(str).str.zfill(5)
            df = df.merge(ref, left_on='ZIP5', right_on='postal_code', how='left')
            def _to_num(x):
                try:
                    return float(str(x).strip().replace(',', '.'))
                except:
                    return np.nan
            df['Latitude']  = pd.to_numeric(df.get('Latitude'), errors='coerce')
            df['Longitude'] = pd.to_numeric(df.get('Longitude'), errors='coerce')
            df['Latitude']  = df['Latitude'].where(df['Latitude'].notna(),  df['latitude'].map(_to_num))
            df['Longitude'] = df['Longitude'].where(df['Longitude'].notna(), df['longitude'].map(_to_num))
            df.drop(columns=['postal_code','latitude','longitude'], inplace=True, errors='ignore')
            lat_col, lon_col = 'Latitude', 'Longitude'
    except Exception as e:
        st.warning(f"基于 ZIP 回填经纬度失败：{e}")

if not lat_col or not lon_col:
    st.error(f"未找到经纬度列，请确认列名包含 {['Latitude','Lat','latitude','lat']} / {['Longitude','Lon','Lng','longitude','lon','lng']}，或提供 Address/ZIP。当前列：{list(df.columns)}")
    st.stop()

def clean_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(',', '.')
    s = re.sub(r'[^0-9\.\-]', '', s)
    try: return float(s)
    except: return np.nan

df['Latitude']  = df[lat_col].apply(clean_num)
df['Longitude'] = df[lon_col].apply(clean_num)

if 'Level' not in df.columns:
    st.error("缺少必要列：Level")
    st.stop()

def to_level(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+)', str(x))
    if not m:
        return np.nan
    v = int(m.group(1))
    return v if 1 <= v <= 7 else np.nan  # 允许 1-7（7 = 网上新增）
df['Level'] = df['Level'].apply(to_level)

# 规范 Level：转为可空整数型，并限定 1..7，其他置为 NA
df['Level'] = pd.to_numeric(df['Level'], errors='coerce').astype('Int64')
df.loc[~df['Level'].between(1, 7), 'Level'] = pd.NA


for need in ['Name', 'Address']:
    if need not in df.columns:
        st.error(f"缺少必要列：{need}")
        st.stop()

zip_candidates = ['ZIP','Zip','zip','ZipCode','ZIP Code','PostalCode','Postal Code','postcode','Postcode','邮编']
zip_col = next((c for c in zip_candidates if c in df.columns), None)
if zip_col is not None:
    df['ZIP'] = df[zip_col]
else:
    df['ZIP'] = df['Address'].apply(_extract_zip_from_text)
df['ZIP']  = df['ZIP'].apply(lambda x: _extract_zip_from_text(x)).astype('string')
df['ZIP5'] = df['ZIP'].str.zfill(5)

try:
    import pgeocode
except ImportError:
    st.error("缺少 pgeocode：请先执行  python -m pip install pgeocode")
    st.stop()

nomi = pgeocode.Nominatim('us')
zip_list = df['ZIP5'].dropna().unique().tolist()
zip_ref = nomi.query_postal_code(zip_list)
zip_ref = zip_ref[['postal_code','latitude','longitude','state_code','place_name','county_name']].dropna(subset=['latitude','longitude'])
zip_ref['postal_code'] = zip_ref['postal_code'].astype(str).str.zfill(5)
df = df.merge(zip_ref, left_on='ZIP5', right_on='postal_code', how='left')

def combine_first_series(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    return a.where(a.notna(), b)

if 'State'  not in df.columns: df['State']  = pd.NA
if 'City'   not in df.columns: df['City']   = pd.NA
if 'County' not in df.columns: df['County'] = pd.NA

df['Latitude']  = combine_first_series(df['latitude'],   df['Latitude']).astype(float)
df['Longitude'] = combine_first_series(df['longitude'],  df['Longitude']).astype(float)
df['State']     = combine_first_series(df['state_code'], df['State']).astype('string')
df['City']      = combine_first_series(df['place_name'], df['City']).astype('string')
df['County']    = combine_first_series(df['county_name'],df['County']).astype('string')
df.drop(columns=['postal_code','latitude','longitude','state_code','place_name','county_name'], inplace=True, errors='ignore')

zip_all = load_zip_all_cached()
cities_master, counties_master = build_city_county_master(zip_all)

# ======================
# 侧边栏筛选
# ======================
with st.sidebar:
    geo_level = st.selectbox("显示范围", ["郡（County）", "城市（City）"], index=0)

    levels_present = sorted([int(x) for x in df['Level'].dropna().unique().tolist()]) or [1,2,3,4,5,6,7]
    level_choice = st.selectbox('选择等级', ['全部'] + levels_present, index=0)

    states_for_level = sorted((counties_master if geo_level.startswith("郡") else cities_master)['State'].unique().tolist())
    state_choice = st.selectbox('选择州 (State)', ['全部'] + states_for_level)

    if geo_level.startswith("郡"):
        units = sorted(counties_master.loc[counties_master['State']==state_choice, 'County'].unique().tolist()) if state_choice!='全部' \
                else sorted(counties_master['County'].unique().tolist())[:5000]
        unit_label = "选择郡 (County)"
    else:
        units = sorted(cities_master.loc[cities_master['State']==state_choice, 'City'].unique().tolist()) if state_choice!='全部' \
                else sorted(cities_master['City'].unique().tolist())[:5000]
        unit_label = "选择城市 (City)"
    unit_choice = st.selectbox(unit_label, ['全部'] + units)

    # “好维修工”与半径阈值
    st.subheader("优选规则")
    good_levels   = st.multiselect("维修工等级", [1,2,3,4,5,6,7], default=[1,2,3,4,5,6])
    radius_miles  = st.slider("半径（英里）", 5, 50, 20, 5)
    min_good      = st.number_input("圈内≥ 好维修工数量", 1, 10, 2, 1)
    only_show_units       = st.checkbox("只显示达标范围", value=True)
    only_show_good_points = st.checkbox("只显示好维修工（按上面的多选）", value=False)


    st.markdown("---")
    # 网上补充相关的筛选（数据源选择 + 过滤）
    source_mode = st.radio("网上补充数据源", ["自动（Google优先）", "只用Google（更快）", "只用OSM（备用，较慢）"], index=0)
    if "hvac_only" not in st.session_state:
        st.session_state.hvac_only = False
    hvac_only = st.checkbox("只看制冷/制热（HVAC）公司", value=st.session_state.hvac_only, key="hvac_only")
    show_only_new = st.checkbox("只看网上新增（Level = 7）", value=False)

    st.markdown("---")
    # 性能设置
    with st.expander("⚡ 性能设置", expanded=False):
        st.checkbox("点位聚合（多点更快）", key="perf_use_cluster", value=st.session_state.get("perf_use_cluster", True))
        st.checkbox("Canvas 渲染矢量", key="perf_prefer_canvas", value=st.session_state.get("perf_prefer_canvas", True))
        st.slider("最多渲染范围数（郡/城市圈）", 200, 5000, int(st.session_state.get("perf_max_units", 1500)), 100, key="perf_max_units")

    
    st.caption(f"🔑 Google Places：{'✅ 已读取' if GOOGLE_PLACES_KEY else '❌ 未设置'}  {_mask_key(GOOGLE_PLACES_KEY)}")

# ======================
# 先计算统计圈（为了把“统计 + 网上补充”移到前面显示）
# ======================
# 基础过滤（不含名称/地址查询）
base_mask = pd.Series(True, index=df.index)
if level_choice != '全部':
    base_mask &= (df['Level'] == level_choice)
if state_choice != '全部':
    base_mask &= (df['State'] == state_choice)
if geo_level.startswith("郡") and 'County' in df.columns and state_choice!='全部' and state_choice is not None:
    pass  # 具体郡/城过滤只影响展示圈集合，统计集合下面再处理
filtered_base = df.loc[base_mask].copy()

if st.session_state.get("hvac_only", False):
    text = (
        filtered_base.get('Name',    pd.Series('', index=filtered_base.index)).astype('string').fillna('') + ' ' +
        filtered_base.get('Address', pd.Series('', index=filtered_base.index)).astype('string').fillna('') + ' ' +
        filtered_base.get('City',    pd.Series('', index=filtered_base.index)).astype('string').fillna('') + ' ' +
        filtered_base.get('County',  pd.Series('', index=filtered_base.index)).astype('string').fillna('')
    )
    filtered_base = filtered_base[
        text.str.contains(HVAC_PAT_STR,  case=False, na=False, regex=True) &
       ~text.str.contains(BLACK_PAT_STR, case=False, na=False, regex=True)
    ]

if geo_level.startswith("郡"):
    base_master = counties_master if state_choice=='全部' else counties_master[counties_master['State']==state_choice]
    name_col = 'County'
    layer_name = "郡圈"
else:
    base_master = cities_master if state_choice=='全部' else cities_master[cities_master['State']==state_choice]
    name_col = 'City'
    layer_name = "城市圈"
if 'unit_choice' in locals() and unit_choice != '全部':
    base_master = base_master[base_master[name_col] == unit_choice]
base_master = base_master.copy()

# 用“已按州/等级/HVAC等筛过”的 filtered_base 参与统计
points_all = filtered_base.dropna(subset=['Latitude','Longitude']).copy()

# “好维修工” = 侧边栏 multiselect（good_levels）
_selected_good_levels = [int(x) for x in (good_levels or [])]
if _selected_good_levels:
    points_good = points_all[points_all['Level'].isin(_selected_good_levels)].copy()
else:
    # 允许把多选清空时，视为没有好维修工
    points_good = points_all.iloc[0:0].copy()


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
            out[i:i+batch] = (dist <= radius_mi).sum(axis=1)
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
                     .head(st.session_state.get("perf_max_units", 1500))
                     .copy())

# ======================
# 统计导出 + 网上补充（移到搜索行之前）
# ======================

unit_key = 'County' if geo_level.startswith("郡") else 'City'
label_total = "统计的郡数" if unit_key == 'County' else "统计的城市数"
label_yes   = "有维修工的郡" if unit_key == 'County' else "有维修工的城市"
label_no    = "没有维修工的郡" if unit_key == 'County' else "没有维修工的城市"

cm_units = base_master.copy()
total_units  = len(cm_units)
covered_units = int(cm_units['meets'].sum())
empty_units   = total_units - covered_units
empty_rate    = (empty_units / total_units) if total_units else 0

left, mid, right, dl = st.columns([0.9, 0.9, 0.9, 1.1])
with left:  st.metric(label_total, f"{total_units:,}")
with mid:   st.metric(label_yes,   f"{covered_units:,}")
with right:
    st.metric(label_no, f"{empty_units:,}（{empty_rate:.1%} 空白率）")


gaps = cm_units[~cm_units['meets']].copy()
gaps['workers_count'] = gaps['good_in_radius'].astype(int)
if unit_key == 'County':
    outcols   = ["State", "County", "ZIP_count", "ZIPs", "cLat", "cLng", "workers_count"]
    sort_cols = ["State", "County"]
else:
    outcols   = ["State", "City", "cLat", "cLng", "workers_count"]
    sort_cols = ["State", "City"]
outcols_present = [c for c in outcols if c in gaps.columns]
gaps_sorted = gaps[outcols_present].sort_values(sort_cols).copy()

if "ZIPs" in gaps_sorted.columns:
    def _z_to_str(z):
        if isinstance(z, (list, tuple, set, np.ndarray)):
            return ", ".join(sorted(map(str, z)))
        return "" if pd.isna(z) else str(z)
    gaps_sorted["ZIPs"] = gaps_sorted["ZIPs"].apply(_z_to_str)

def _build_xlsx(df_to_export: pd.DataFrame, sheet_name="Sheet1") -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_to_export.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf

tag    = f"r{int(radius_miles)}_min{int(min_good)}"
prefix = "counties" if unit_key == "County" else "cities"
fname  = f"{prefix}_not_meeting_threshold_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
excel_bytes = _build_xlsx(gaps_sorted, sheet_name="EmptyUnits")

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

# 🌐 网上补充（折叠隐藏，保持原位置但已上移到这里）
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_osm_overpass(lat, lng, radius_m=30000, hvac_only=False):
    if hvac_only:
        q = f"""
[out:json][timeout:25];
(
  node(around:{radius_m},{lat},{lng})["craft"="hvac"];
  way(around:{radius_m},{lat},{lng})["craft"="hvac"];
  node(around:{radius_m},{lat},{lng})["name"~"hvac|air.?conditioning|heating|cooling|refrigeration|furnace|boiler", i];
  way(around:{radius_m},{lat},{lng})["name"~"hvac|air.?conditioning|heating|cooling|refrigeration|furnace|boiler", i];
);
out center tags 60;
"""
    else:
        q = f"""
[out:json][timeout:25];
(
  node(around:{radius_m},{lat},{lng})["craft"~"hvac|electrician|plumber"];
  way(around:{radius_m},{lat},{lng})["craft"~"hvac|electrician|plumber"];
);
out center tags 60;
"""
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=q.encode("utf-8"), timeout=30)
        j = r.json()
    except Exception:
        return []

    out = []
    for e in j.get("elements", []):
        tags = e.get("tags", {}) or {}
        name = tags.get("name") or tags.get("brand")
        if not name:
            continue
        lat_e = e.get("lat") or (e.get("center") or {}).get("lat")
        lon_e = e.get("lon") or (e.get("center") or {}).get("lon")
        if lat_e is None or lon_e is None:
            continue
        addr = ", ".join(filter(None, [
            tags.get("addr:housenumber"), tags.get("addr:street"),
            tags.get("addr:city"), tags.get("addr:state"), tags.get("addr:postcode")
        ]))
        text = f"{name} {addr}"
        if BLACK_RE.search(text):
            continue
        if st.session_state.get("hvac_only", False) and (not HVAC_RE.search(text) and tags.get("craft") != "hvac"):
            continue
        out.append({
            "Name": name,
            "Address": addr if addr else pd.NA,
            "Latitude": float(lat_e),
            "Longitude": float(lon_e),
            "Level": np.nan,
            "ZIP": tags.get("addr:postcode"),
            "State": pd.NA, "City": pd.NA, "County": pd.NA,
            "Source": "web-osm",
            "SourceTag": tags.get("craft",""),
            "SourceId": str(e.get("id")),
            "Rating": pd.NA,
            "UserRatingsTotal": pd.NA,
        })
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def _gplaces_nearby_once(params, log=None):
    base = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    try:
        r = requests.get(base, params=params, timeout=20)
        j = r.json()
    except Exception as e:
        if log: log(f"Google 请求异常：{e}")
        return None
    status = j.get("status", "")
    if status != "OK":
        msg = j.get("error_message", "")
        if log: log(f"Google 返回：{status} {(' - ' + msg) if msg else ''}")
        return {"status": status, "results": j.get("results", []), "next_page_token": j.get("next_page_token")}
    return {"status": "OK", "results": j.get("results", []), "next_page_token": j.get("next_page_token")}

@st.cache_data(show_spinner=False, ttl=900)
def _cache_ok_signature(params_signature, page):
    return True

def fetch_google_places(lat, lng, radius_m=30000, api_key=None, hvac_only=False, log=None):
    key = api_key if api_key is not None else GOOGLE_PLACES_KEY
    if not key:
        if log: log("未检测到 Google API Key，跳过 Google。")
        return []
    params = {"key": key, "location": f"{lat},{lng}", "radius": int(radius_m), "type": "hvac_contractor", "language": "en"}
    out, token, page = [], None, 0
    while True:
        req = dict(params)
        if token:
            req["pagetoken"] = token
        resp = _gplaces_nearby_once(req, log=log)
        if not resp: break
        if resp["status"] != "OK": break
        _cache_ok_signature(tuple(sorted(req.items())), page)
        for it in resp["results"]:
            name = it.get("name", "") or ""
            addr = it.get("vicinity") or it.get("formatted_address") or ""
            types = it.get("types", []) or []
            text  = f"{name} {addr}"
            is_hvac_type = ("hvac_contractor" in types)
            is_hvac_text = bool(HVAC_RE.search(text))
            if hvac_only and (not (is_hvac_type or is_hvac_text)): continue
            if BLACK_RE.search(text): continue
            out.append({
                "Name": name, "Address": addr,
                "Latitude": it["geometry"]["location"]["lat"], "Longitude": it["geometry"]["location"]["lng"],
                "Level": np.nan, "ZIP": pd.NA, "State": pd.NA, "City": pd.NA, "County": pd.NA,
                "Source": "web-google", "SourceTag": "hvac_contractor" if is_hvac_type else "",
                "SourceId": it.get("place_id"), "Rating": it.get("rating"), "UserRatingsTotal": it.get("user_ratings_total"),
            })
        token = resp.get("next_page_token")
        page += 1
        if not token or page >= 3: break
        time.sleep(2.2)  # 官方建议 pagetoken 至少等 ~2s

    if hvac_only and not out:
        if log: log("type=hvac_contractor 未命中，使用 keyword 兜底…")
        kw = 'hvac OR "air conditioning" OR heating OR "heat pump" OR furnace OR boiler OR chiller'
        req_kw = {"key": key, "location": f"{lat},{lng}", "radius": int(radius_m), "keyword": kw, "language": "en"}
        resp = _gplaces_nearby_once(req_kw, log=log)
        if resp and resp.get("results"):
            for it in resp["results"]:
                name = it.get("name", "") or ""
                addr = it.get("vicinity") or it.get("formatted_address") or ""
                text = f"{name} {addr}"
                if BLACK_RE.search(text): continue
                if not HVAC_RE.search(text): continue
                out.append({
                    "Name": name, "Address": addr,
                    "Latitude": it["geometry"]["location"]["lat"], "Longitude": it["geometry"]["location"]["lng"],
                    "Level": np.nan, "ZIP": pd.NA, "State": pd.NA, "City": pd.NA, "County": pd.NA,
                    "Source": "web-google", "SourceTag": "keyword",
                    "SourceId": it.get("place_id"), "Rating": it.get("rating"), "UserRatingsTotal": it.get("user_ratings_total"),
                })
    if log: log(f"Google 返回有效条数：{len(out)}")
    return out

def fetch_online_candidates_for_county(row, radius_m=30000, api_key=None, hvac_only=False, source_mode="自动（Google优先）", log=None):
    lat = float(row["cLat"]); lng = float(row["cLng"])
    items = []
    if source_mode == "只用Google（更快）":
        items = fetch_google_places(lat, lng, radius_m=radius_m, api_key=api_key, hvac_only=hvac_only, log=log)
    elif source_mode == "只用OSM（备用，较慢）":
        items = fetch_osm_overpass(lat, lng, radius_m=radius_m, hvac_only=hvac_only)
    else:
        if api_key:
            items = fetch_google_places(lat, lng, radius_m=radius_m, api_key=api_key, hvac_only=hvac_only, log=log)
        if not items:
            items = fetch_osm_overpass(lat, lng, radius_m=radius_m, hvac_only=hvac_only)
    for it in items:
        it["State"]  = row["State"]
        it["County"] = row.get("County", pd.NA)
        it["City"]   = row.get("City",   pd.NA)
    return items

with st.expander("🌐 网上补充数据", expanded=False):
    st.caption("（按当前选择的州/郡/城市抓取）")
    col_a, col_b, col_c = st.columns([1.2, 1, 1.2])
    with col_a:
        max_cnties = st.number_input("最多抓取的范围数量", 1, 200, 10, 1)
    with col_b:
        search_rad = st.slider("抓取半径（公里）", 5, 80, 30, 5)
    with col_c:
        merge_back = st.checkbox("抓取后合并到数据集中（并加入地图）", value=True)
    do_fetch = st.button("🔵 一键补充网上维修工（蓝色图标）", use_container_width=True)
    online_df = pd.DataFrame()
    if do_fetch:
        log_box = st.empty()
        def log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            log_box.markdown(f"📝 **[{ts}]** {msg}")
        take_n = int(max_cnties); radius_m_fetch = int(search_rad * 1000)
        base = counties_master if geo_level.startswith("郡") else cities_master
        if unit_choice != '全部' and state_choice != '全部':
            rows = base[(base["State"] == state_choice) & (base[name_col] == unit_choice)].head(1).to_dict("records")
        elif state_choice != '全部':
            rows = base[base["State"] == state_choice].sort_values(["State", name_col]).head(take_n).to_dict("records")
        else:
            rows = base.sort_values(["State", name_col]).head(take_n).to_dict("records")
        if not rows:
            st.warning("未选中任何抓取对象。请在侧边栏选择州/郡/城市后再试。")
        else:
            log(f"开始抓取 {len(rows)} 个范围（半径 {search_rad}km，HVAC仅限：{st.session_state.get('hvac_only', False)}，数据源：{source_mode}）…")
            all_items = []
            for i, r0 in enumerate(rows, 1):
                unit_name = r0.get("County") if geo_level.startswith("郡") else r0.get("City")
                log(f"第 {i}/{len(rows)} 个：{r0['State']} / {unit_name} —— 请求中…")
                try:
                    items = fetch_online_candidates_for_county(
                        r0, radius_m=radius_m_fetch, api_key=GOOGLE_PLACES_KEY,
                        hvac_only=st.session_state.get('hvac_only', False), source_mode=source_mode, log=log)
                    all_items.extend(items)
                    log(f"✓ 新增 {len(items)} 条，累计 {len(all_items)} 条。")
                except Exception as e:
                    log(f"× 抓取失败：{e}")
                time.sleep(0.2)
            if all_items:
                online_df = (pd.DataFrame(all_items)
                             .drop_duplicates(subset=["Source","SourceId"], keep="first")
                             .reset_index(drop=True))
                online_df["Level"] = 7
            else:
                online_df = pd.DataFrame(columns=["Name","Address","Latitude","Longitude","Level","State","City","County","ZIP","Source","SourceId","Rating","UserRatingsTotal"])
            if not online_df.empty:
                add_fg = folium.FeatureGroup(name="网上新增维修工（抓取）", show=True)
                # 延后 add 到 map，等 map 创建后再挂
                st.session_state["_web_new_layer"] = online_df.to_dict("records")
            buf_new = io.BytesIO()
            with pd.ExcelWriter(buf_new, engine="openpyxl") as w:
                online_df.to_excel(w, index=False, sheet_name="WebNew")
            buf_new.seek(0)
            st.download_button("下载本次抓取清单（Excel）", data=buf_new,
                               file_name=f"web_new_workers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True, key="dl_web_new")
            if online_df.empty:
                st.warning("抓取完成，但 0 条有效结果（可能 API 未返回或被过滤，或未设置 Google Key）。")
            else:
                st.success(f"抓取完成，共 {len(online_df)} 条有效结果。")
            if merge_back and not online_df.empty:
                cols = ["Name","Address","Latitude","Longitude","Level","State","City","County","ZIP"]
                for c in cols:
                    if c not in online_df.columns: online_df[c] = pd.NA
                st.session_state.df = pd.concat([st.session_state.df, online_df[cols]], ignore_index=True)
                st.toast("已把网上新增点（Level=7）合并到数据集中。")
                _safe_rerun()


# ======================
# 搜索行（对调后：现在紧贴地图上方）
# ======================
# 注意：客户地址支持 on_change 回车定位；不再有“定位”按钮
c1, c2, c3 = st.columns([0.28, 0.36, 0.36])
with c1:
    q_name = st.text_input("维修工名称", key="q_name", placeholder="例如：ACME Tech", autocomplete="off")
with c2:
    q_addr = st.text_input("维修工地址关键词", key="q_addr", placeholder="城市/州/街道/ZIP", autocomplete="off")
with c3:
    st.text_input("客户地址（回车定位📍）", key="cust_quick_addr",
                  placeholder="输入后按回车定位/或输入坐标 34.0522,-118.2437",
                  on_change=_on_cust_addr_change)

# 弹出透明提示（不常驻）
if "_last_cust_geocode_msg" in st.session_state:
    st.toast(st.session_state.pop("_last_cust_geocode_msg"))

# ======================
# 过滤（基础 + HVAC） + 顶部命中集合
# ======================
def _s(val):
    try:
        if val is None: return ""
        if isinstance(val, float) and np.isnan(val): return ""
        import pandas as pd
        if pd.isna(val): return ""
    except Exception:
        pass
    return str(val)

def _full_address_from_row(row):
    def _clean(v):
        try:
            import pandas as pd
            if v is None or pd.isna(v):
                return ""
        except Exception:
            if v is None:
                return ""
        return str(v).strip()

    addr = _clean(row.get('Address', ''))
    if addr:
        return addr

    city  = _clean(row.get('City', ''))
    state = _clean(row.get('State', ''))
    zip5  = _clean(row.get('ZIP', ''))
    parts = [p for p in (city, state, zip5) if p]
    return ", ".join(parts) if parts else ""



mask = pd.Series(True, index=df.index)
if level_choice != '全部':
    mask &= (df['Level'] == level_choice)
if state_choice != '全部':
    mask &= (df['State'] == state_choice)
if geo_level.startswith("郡") and unit_choice != '全部':
    mask &= (df['County'] == unit_choice)
if (not geo_level.startswith("郡")) and unit_choice != '全部':
    mask &= (df['City'] == unit_choice)

filtered = df.loc[mask].copy()

if st.session_state.get("hvac_only", False):
    text = (
        filtered.get('Name',    pd.Series('', index=filtered.index)).astype('string').fillna('') + ' ' +
        filtered.get('Address', pd.Series('', index=filtered.index)).astype('string').fillna('') + ' ' +
        filtered.get('City',    pd.Series('', index=filtered.index)).astype('string').fillna('') + ' ' +
        filtered.get('County',  pd.Series('', index=filtered.index)).astype('string').fillna('')
    )
    filtered = filtered[
        text.str.contains(HVAC_PAT_STR,  case=False, na=False, regex=True) &
       ~text.str.contains(BLACK_PAT_STR, case=False, na=False, regex=True)
    ]

# 应用“只看新增/只看好维修工”
points = filtered.dropna(subset=['Latitude','Longitude']).copy()
if show_only_new:
    points = points[points['Level'].eq(7)]

# ✅ 用侧边栏的 good_levels 来过滤地图点位，而不是写死 1..6
if only_show_good_points:
    _sel_lvls = [int(x) for x in (good_levels or [])]
    points = points[points['Level'].isin(_sel_lvls)] if _sel_lvls else points.iloc[0:0]

# 命中集合（用于 🚩）
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
        _contains_safe(points.get('ZIP',    pd.Series("", index=points.index)),   q_addr) |
        _contains_safe(points.get('State',  pd.Series("", index=points.index)),   q_addr)
    )
    matched = matched[addr_mask]
search_active = bool(has_query) and (len(matched) > 0)

# ======================
# 地图绘制
# ======================
US_STATES_GEO_PATH = os.path.join(data_dir, "us_states.geojson")
# 统一颜色表（1–7）：1=绿, 2=金, 3=红, 4=粉, 5=紫, 6=黑, 7=蓝(网上补充)
LEVEL_COLORS = {
    1: '#2ecc71',
    2: '#FFD700',
    3: '#FF4D4F',
    4: '#FFC0CB',
    5: '#8A2BE2',
    6: '#000000',
    7: '#1E90FF',
}


prefer_canvas = st.session_state.get("perf_prefer_canvas", True)
m = folium.Map(location=[37.8, -96.0], zoom_start=4, keyboard=False,
               prefer_canvas=prefer_canvas, tiles="CartoDB positron")
m.get_root().header.add_child(folium.Element("""
<style>
.leaflet-container:focus, .leaflet-container:focus-visible { outline: none !important; }
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
        return {'fillColor':'#ffffff','fillOpacity':0.0,'color':'#2563eb','weight':3.0 if is_selected else 1.8,'dashArray':None}
    gj = folium.GeoJson(
        data=states_fc, name="US States",
        style_function=style_fn, highlight_function=None,
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'])
    ).add_to(m)

    if state_choice != '全部':
        target = next((f for f in feats if (f.get('id') or f.get('properties', {}).get('state_code')) == state_choice), None)
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

# 郡/城市圈
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
    folium.Circle(location=[r['cLat'], r['cLng']], radius=radius_m,
                  color=ring_color, weight=1.6, fill=True, fill_opacity=0.05,
                  tooltip=tip).add_to(unit_fg)

# ---- 弹窗模板（更宽 + 每项单独一行）----
POPUP_MAX_W = 520
# 替换你当前的 make_worker_popup 定义为下方版本
def make_worker_popup(name, level, address=None, zip_code=None, distance_text="", **kwargs):
    """
    兼容旧调用：如果传了 state=... 也不报错；
    优先用 address；没有就回退用 state。
    """
    # 兼容旧参数名
    if not address:
        address = kwargs.get("state", "")  # 让旧代码不崩

    html = f"""
    <div style="min-width:420px; font-size:13px; line-height:1.4; white-space:normal;">
      <div><b>名称：</b>{_s(name)}</div>
      <div><b>等级：</b>{_s(level)}</div>
      <div><b>地址：</b>{_s(address)}</div>
      <div><b>ZIP：</b>{_s(zip_code)}</div>
      <div><b>距离：</b>{_s(distance_text)}</div>
    </div>
    """
    return folium.Popup(html, max_width=POPUP_MAX_W)


# 距离/时间工具
def haversine_miles(lat1, lng1, lat2, lng2):
    R = 3958.7613
    p1 = np.radians([lat1, lng1]); p2 = np.radians([lat2, lng2])
    dphi = p2[0]-p1[0]; dlmb = p2[1]-p1[1]
    a = np.sin(dphi/2)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(dlmb/2)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def osrm_drive_info(lat1, lng1, lat2, lng2, timeout=8):
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{lng1},{lat1};{lng2},{lat2}"
        r = requests.get(url, params={"overview":"false"}, timeout=timeout)
        j = r.json()
        if j.get("code") == "Ok" and j.get("routes"):
            dist_m = j["routes"][0]["distance"]; dur_s = j["routes"][0]["duration"]
            return dist_m/1609.34, dur_s/60.0
    except Exception:
        pass
    return None

cust_pin = st.session_state.get("cust_quick_pin")

# 点位层
workers_fg = folium.FeatureGroup(name="维修工点位", show=True).add_to(m)
use_cluster = st.session_state.get("perf_use_cluster", True)

def popup_distance_text(lat, lng, prefer_drive=False):
    if not cust_pin:
        return "-"
    dline = haversine_miles(cust_pin["lat"], cust_pin["lng"], float(lat), float(lng))
    if prefer_drive:
        drive = osrm_drive_info(cust_pin["lat"], cust_pin["lng"], float(lat), float(lng))
        if drive:
            return f"{drive[0]:.1f} mi · {int(round(drive[1]))} min"
    return f"{dline:.1f} mi（直线）"

if use_cluster and len(points) > 2000:
    clusters = {}
    for lvl, col in sorted(LEVEL_COLORS.items()):
        clusters[lvl] = MarkerCluster(
            name=f"Level {lvl}",
            icon_create_function=f"""
            function(cluster) {{
              var count = cluster.getChildCount();
              return new L.DivIcon({{
                html: '<div style="background:{col};opacity:0.85;border-radius:20px;width:36px;height:36px;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;border:2px solid white;">'+count+'</div>',
                className: 'marker-cluster', iconSize: new L.Point(36, 36)
              }});
            }}
            """
        ).add_to(workers_fg)
    # …下面 add marker 时颜色也统一取 LEVEL_COLORS …

    for _, row in points.iterrows():
        lvl = int(row['Level']) if not pd.isna(row['Level']) else None
        color = LEVEL_COLORS.get(lvl, '#3388ff')
        distance_text = popup_distance_text(row['Latitude'], row['Longitude'], prefer_drive=False)
        popup_obj = make_worker_popup(
            name=row.get('Name',''),
            level=row.get('Level',''),
            state=row.get('State',''),
            zip_code=row.get('ZIP',''),
            distance_text=distance_text
        )
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup=popup_obj
        ).add_to(clusters.get(lvl, workers_fg))
else:
    for _, row in points.iterrows():
        lvl = int(row['Level']) if not pd.isna(row['Level']) else None
        color = LEVEL_COLORS.get(lvl, '#3388ff')
        distance_text = popup_distance_text(row['Latitude'], row['Longitude'], prefer_drive=False)
        popup_obj = make_worker_popup(
    name=row.get('Name',''),
    level=row.get('Level',''),
    address=_full_address_from_row(row),
    zip_code=row.get('ZIP',''),
    distance_text=distance_text
)
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup=popup_obj
        ).add_to(workers_fg)

# 搜索命中：红旗（优先尝试驾车距离/时间）
if search_active and len(matched) > 0:
    for _, r in matched.iterrows():
        dist_txt = popup_distance_text(r['Latitude'], r['Longitude'], prefer_drive=True)
        popup_obj = make_worker_popup(
            name=r.get('Name',''),
            level=r.get('Level',''),
            address=_full_address_from_row(r),
            zip_code=r.get('ZIP',''),
            distance_text=dist_txt
        )
        folium.Marker(
            location=[float(r['Latitude']), float(r['Longitude'])],
            icon=big_flag_icon(size_px=42),
            tooltip=f"🔎 命中：{r.get('Name','')}",
            popup=popup_obj,
            z_index_offset=10000
        ).add_to(m)

# 客户快速定位点（与维修工图标可同时存在）
if cust_pin:
    p = cust_pin
    folium.Marker(
        location=[float(p["lat"]), float(p["lng"])],
        icon=customer_pin_icon(size_px=42),
        tooltip=f"📍 {p.get('formatted','客户地址')}",
        popup=f"<b>客户地址：</b>{_s(p.get('formatted'))}",
        z_index_offset=12000
    ).add_to(m)

# 若有“网上新增层”缓存，画出来
if "_web_new_layer" in st.session_state:
    add_fg = folium.FeatureGroup(name="网上新增维修工（抓取）", show=True).add_to(m)
    for r in st.session_state.pop("_web_new_layer"):
        name = _s(r.get("Name")); addr = _s(r.get("Address")); rating = _s(r.get("Rating"))
        dist_txt = popup_distance_text(r.get("Latitude"), r.get("Longitude"), prefer_drive=False)
        popup_obj = make_worker_popup(name, r.get("Level","7"), _s(r.get("Address","")), r.get("ZIP",""), dist_txt)

        folium.Marker(
            location=[float(r["Latitude"]), float(r["Longitude"])],
            icon=blue_wrench_icon(),
            tooltip=f"🔧 新增：{name}",
            popup=popup_obj
        ).add_to(add_fg)

# 初始/智能缩放
CONUS_BOUNDS = [[24.5, -125.0], [49.5, -66.9]]
def fit_initial_or_search(map_obj, nat_bnds, state_bnds, matched_df, search_active):
    if "_zoom_bounds" in st.session_state:
        map_obj.fit_bounds(st.session_state.pop("_zoom_bounds"))
        return
    if search_active and matched_df is not None and len(matched_df) > 0:
        if len(matched_df) == 1:
            lat = float(matched_df['Latitude'].iloc[0]); lng = float(matched_df['Longitude'].iloc[0])
            pad = 0.35; b = [[lat - pad, lng - pad], [lat + pad, lng + pad]]
        else:
            b = [[matched_df['Latitude'].min(),  matched_df['Longitude'].min()],
                 [matched_df['Latitude'].max(),  matched_df['Longitude'].max()]]
        map_obj.fit_bounds(b); return
    if (state_choice != '全部') and state_bnds:
        map_obj.fit_bounds(state_bnds); return
    map_obj.fit_bounds(nat_bnds)

fit_initial_or_search(m, CONUS_BOUNDS, selected_bounds, matched if 'matched' in locals() else None, search_active)

# 右上角图例
lvl_order = sorted(LEVEL_COLORS.keys())
lvl_counts = {lvl: int(points['Level'].eq(lvl).sum()) for lvl in lvl_order}

rows_html = "".join([
    f"<span><span style='display:inline-block;width:10px;height:10px;background:{LEVEL_COLORS[lvl]};"
    f"margin-right:6px;border-radius:2px;{'border:1px solid #eee;' if lvl==6 else ''}'></span>{lvl}</span>"
    f"<span>{lvl_counts.get(lvl,0)}</span>"
    for lvl in lvl_order
])

legend_html = f"""
<div style="
  position: fixed; top: 10px; right: 12px; z-index: 9999;
  background: #fff; padding: 6px 8px; border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,.20); font-size: 12px; line-height: 1.4;
  width: max-content; max-width: 220px; border: 1px solid rgba(0,0,0,.08);
">
  <div style="font-weight:600; margin-bottom:4px;">等级颜色</div>
  <div style="display:grid; grid-template-columns: auto auto; grid-column-gap:8px; grid-row-gap:4px; align-items:center;">
    {rows_html}
  </div>
  <div style="margin-top:6px; border-top:1px dashed #e5e7eb; padding-top:6px; font-weight:600;">
    总数：{int(points.shape[0])}
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))


# 图层开关
folium.LayerControl(collapsed=True, position='topleft').add_to(m)

# 渲染地图（靠上显示）
if USE_STATIC_MAP:
    from streamlit.components.v1 import html
    html(m.get_root().render(), height=760)
else:
    map_height = st.session_state.get("map_height", 760)
    st_folium(m, use_container_width=True, height=map_height)