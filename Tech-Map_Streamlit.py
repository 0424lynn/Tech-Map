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
from folium.plugins import MarkerCluster, BeautifyIcon

# ======================
# Basic
# ======================
st.set_page_config(page_title="Tech Map", layout="wide")
USE_STATIC_MAP = True  # Folium raw HTML (faster)

def _safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ---------- Icons ----------
def ring_pin_icon(color_hex: str = "#1E90FF", size_h_px: int = 38):
    w = int(size_h_px * 0.75)
    html = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{size_h_px}" viewBox="0 0 48 64"
         style="filter: drop-shadow(0 1px 2px rgba(0,0,0,.35));">
      <path d="M24 2C12 2 2 12.5 2 25c0 17 22 37 22 37s22-20 22-37C46 12.5 36 2 24 2z"
            fill="{color_hex}"/>
      <circle cx="24" cy="25" r="10" fill="#ffffff"/>
    </svg>
    """
    return folium.DivIcon(html=html, icon_size=(w, size_h_px), icon_anchor=(w // 2, int(size_h_px * 0.92)))

def blue_wrench_icon(size_px: int = 24):
    return ring_pin_icon("#8B4513", size_h_px=max(22, int(size_px * 1.2)))

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
    return folium.DivIcon(
        html=f"""
        <div style="filter: drop-shadow(0 0 2px rgba(0,0,0,.35));">
          <span style="font-size:{size_px}px; line-height:1;">📍</span>
        </div>
        """,
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, int(size_px * 0.92)),
    )

def customer_house_icon(size_px: int = 24):
    return folium.DivIcon(
        html=f"""
        <div style="filter: drop-shadow(0 0 2px rgba(0,0,0,.35));">
          <span style="font-size:{size_px}px; line-height:1;">🏠</span>
        </div>
        """,
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, int(size_px * 0.90)),
    )
    

# ---------- Read Google Key ----------
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
# Global Styles
# ======================
st.markdown("""
<style>
div[data-testid="stDecoration"]{display:none!important;}
header[data-testid="stHeader"]{ height:2.4rem !important; visibility:visible !important; }
:root, .stApp { --top-toolbar-height:2.4rem !important; }
button[title="Toggle sidebar"]{ opacity:1 !important; pointer-events:auto !important; }

.stAppViewContainer{ padding-top:0!important; }
.main .block-container{ padding-top:.1rem!important; margin-top:0!important; }
.main .block-container > div{ margin-top:.3rem!important; }

div[data-testid="stHorizontalBlock"]{ margin-bottom:.1rem!important; }

html, body, .stApp, .main .block-container { font-size: 13px !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { font-size: 12.5px !important; }

/* 指标数字调小 */
[data-testid="stMetricValue"] { font-size: 16px !important; line-height:1.1!important; font-weight:700!important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; line-height:1.1!important; }
[data-testid="stMetricDelta"] { font-size: 10px !important; line-height:1.1!important; }
div[data-testid="stMetric"] > div { padding-top:2px!important; padding-bottom:2px!important; margin:0!important; }

div[data-testid="stIFrame"]{ margin-top: .1rem!important; }

.stDownloadButton > button{
  background:#2563eb!important; color:#fff!important;
  border:1px solid #1d4ed8!important; border-radius:8px!important;
  box-shadow:0 2px 6px rgba(37,99,235,.25)!important;
}
.stDownloadButton > button:hover { background-color:#1e40af!important; border-color:#1e3a8a!important; }
.stDownloadButton > button:active { background-color:#1d4ed8!important; transform: translateY(0.5px); }

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
# Data dirs
# ======================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = r"C:\Users\jeffy\chris\tech map"
DATA_DIR_ENV = os.getenv("TECH_MAP_DATA_DIR")
DATA_DIR_DEFAULT = DATA_DIR_ENV or (LOCAL_DIR if os.path.exists(LOCAL_DIR) else os.path.join(APP_DIR, "data"))
os.makedirs(DATA_DIR_DEFAULT, exist_ok=True)
SUPPORT_EXTS = (".csv", ".xlsx", ".xls")

if "data_dir_path" not in st.session_state:
    st.session_state.data_dir_path = DATA_DIR_DEFAULT

data_dir = st.session_state.data_dir_path
os.makedirs(data_dir, exist_ok=True)

def _list_tech_files():
    try:
        files = []
        for f in os.listdir(data_dir):
            fl = f.lower()
            if not fl.endswith(SUPPORT_EXTS):
                continue
            if any(k in fl for k in ["customer", "customers", "cust_", "cust-", "_cust"]):
                continue
            files.append(f)
        return sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    except Exception:
        return []

def _list_customer_files():
    try:
        files = [f for f in os.listdir(data_dir)
                 if f.lower().endswith(SUPPORT_EXTS)
                 and any(k in f.lower() for k in ["customer","customers","cust_","cust-","_cust"])]
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

def _save_uploaded(uploaded, folder: str, prefix: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(uploaded.name)
    if prefix:
        base = f"{prefix}_{base}"
    fpath = os.path.join(folder, f"{base}_{ts}{ext}")
    with open(fpath, "wb") as f:
        f.write(uploaded.read())
    return fpath

# ======================
# Session init
# ======================
if "df" not in st.session_state:
    st.session_state.df = None
if "data_meta" not in st.session_state:
    st.session_state.data_meta = {}
if "cust_df" not in st.session_state:
    st.session_state.cust_df = None
if "cust_meta" not in st.session_state:
    st.session_state.cust_meta = {}

# 自动加载最新技师文件（排除客户）
_files = _list_tech_files()
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
        st.error(f"Failed to load {_files[0]}: {e}")

# 自动加载最新客户文件
_cfiles = _list_customer_files()
if st.session_state.cust_df is None and _cfiles:
    c_latest = os.path.join(data_dir, _cfiles[0])
    try:
        st.session_state.cust_df = _load_df(c_latest)
        st.session_state.cust_meta = {
            "filename": _cfiles[0],
            "path": c_latest,
            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": len(st.session_state.cust_df)
        }
    except Exception as e:
        st.warning(f"Customers auto-load failed: {e}")

df = st.session_state.get("df", None)
if df is None:
    # Sidebar data source（首次无数据时）
    with st.sidebar:
        st.markdown("---")
        with st.expander("📁 Data Source (fixed folder)", expanded=True):
            new_dir = st.text_input("Data folder path", value=st.session_state.data_dir_path)
            if new_dir != st.session_state.data_dir_path:
                st.session_state.data_dir_path = new_dir
            os.makedirs(st.session_state.data_dir_path, exist_ok=True)

            st.markdown("**Technicians**")
            files2 = _list_tech_files()
            if files2:
                pick = st.selectbox("Pick saved tech file", files2, index=0)
                if st.button("Load selected (tech)"):
                    try:
                        path = os.path.join(st.session_state.data_dir_path, pick)
                        st.session_state.df = _load_df(path)
                        st.session_state.data_meta = {
                            "filename": pick, "path": path,
                            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.success(f"Loaded: {pick}")
                        _safe_rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")

            new_file = st.file_uploader("Upload tech data (save into folder)", type=['csv','xlsx','xls'])
            if new_file is not None:
                try:
                    saved_path = _save_uploaded(new_file, st.session_state.data_dir_path, prefix="TECH")
                    st.session_state.df = _load_df(saved_path)
                    st.session_state.data_meta = {
                        "filename": os.path.basename(saved_path), "path": saved_path,
                        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success(f"Uploaded & loaded: {os.path.basename(saved_path)}")
                    _safe_rerun()
                except Exception as e:
                    st.error(f"Upload/read failed: {e}")

            st.markdown("---")
            st.markdown("**Customers**")
            cfiles = _list_customer_files()
            if cfiles:
                cpick = st.selectbox("Pick saved customer file", cfiles, index=0, key="pick_cust_file_top")
                if st.button("Load selected (customers)", key="btn_load_cust_top"):
                    try:
                        cpath = os.path.join(st.session_state.data_dir_path, cpick)
                        st.session_state.cust_df = _load_df(cpath)
                        st.session_state.cust_meta = {
                            "filename": cpick, "path": cpath,
                            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "rows": len(st.session_state.cust_df)
                        }
                        st.success(f"Customers loaded: {cpick}")
                        _safe_rerun()
                    except Exception as e:
                        st.error(f"Load customers failed: {e}")

            cust_file = st.file_uploader("Upload customers (save into folder)", type=['csv','xlsx','xls'], key="cust_uploader_top")
            if cust_file is not None:
                try:
                    c_saved = _save_uploaded(cust_file, st.session_state.data_dir_path, prefix="CUSTOMERS")
                    cdf = _load_df(c_saved)
                    st.session_state.cust_df = cdf
                    st.session_state.cust_meta = {
                        "filename": os.path.basename(c_saved), "path": c_saved,
                        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rows": len(cdf)
                    }
                    st.success(f"Customers uploaded & loaded: {os.path.basename(c_saved)}")
                    _safe_rerun()
                except Exception as e:
                    st.error(f"Customers upload/read failed: {e}")

    st.warning("No technician dataset loaded. Use the left panel (**📁 Data Source**) to pick or upload a file.")
    st.stop()

# ======================
# ==== PERF：全局性能参数（可通过侧栏调节） ====
# ======================
DEFAULT_MAX_TECH_MARKERS = 500
DEFAULT_MAX_CUST_MARKERS = 200
DEFAULT_NEAR_PIN_BOOST_MI = 60
DEFAULT_ZOOM_GRID_DECIMALS = (2, 3)

# ==== PERF：抽稀工具函数 ====
def _haversine_batch(lat1, lng1, lat_arr, lng_arr):
    R = 3958.7613
    p1 = np.radians([lat1, lng1])
    p2 = np.radians(np.vstack([lat_arr, lng_arr]).T)
    dphi = p2[:,0] - p1[0]
    dlmb = p2[:,1] - p1[1]
    a = np.sin(dphi/2)**2 + np.cos(p1[0])*np.cos(p2[:,0])*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))  # miles

def _priority_mask(df_points, matched_idx, inhouse_mask, cust_pin, near_radius_mi):
    keep = pd.Series(False, index=df_points.index)
    if matched_idx is not None and len(matched_idx) > 0:
        keep.loc[matched_idx] = True
    if inhouse_mask is not None and inhouse_mask.any():
        keep = keep | inhouse_mask
    if cust_pin:
        lat = df_points['Latitude'].to_numpy(dtype=float)
        lng = df_points['Longitude'].to_numpy(dtype=float)
        d = _haversine_batch(cust_pin['lat'], cust_pin['lng'], lat, lng)
        keep = keep | (d <= near_radius_mi)
    return keep

def _grid_key(lat, lng, decimals):
    return np.round(lat, decimals=decimals).astype(np.float32), np.round(lng, decimals=decimals).astype(np.float32)

def thin_points(df_points, target_max, matched_idx=None, inhouse_mask=None, cust_pin=None,
                near_radius_mi=DEFAULT_NEAR_PIN_BOOST_MI, decimals_candidates=DEFAULT_ZOOM_GRID_DECIMALS):
    if len(df_points) <= target_max:
        return df_points.index
    keep_mask = _priority_mask(df_points, matched_idx, inhouse_mask, cust_pin, near_radius_mi)
    keep_idx = df_points.index[keep_mask]
    remain = df_points.index[~keep_mask]
    if len(keep_idx) >= target_max:
        return keep_idx[:target_max]
    quota = target_max - len(keep_idx)
    rest_df = df_points.loc[remain]
    chosen = []
    for dec in decimals_candidates + (4, 5):
        latg, lngg = _grid_key(rest_df['Latitude'].to_numpy(float),
                               rest_df['Longitude'].to_numpy(float), dec)
        grid = pd.DataFrame({'i': rest_df.index, 'latg': latg, 'lngg': lngg})
        sampled = grid.groupby(['latg','lngg'], sort=False).head(1)['i'].tolist()
        chosen = sampled
        if len(chosen) >= quota * 0.95:
            break
    chosen = chosen[:quota]
    return pd.Index(keep_idx.tolist() + chosen)

def should_disable_popups(n_points, cutoff=2800):
    return n_points > cutoff

# ======================
# Cached refs & helpers
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
                .agg(cLat=('latitude','mean'), cLng=('longitude','mean'), ZIP_count=('postal_code','nunique')))
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

# Address helpers
LATLNG_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$')
_US_STATES = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR"}

def _smart_zip_from_text(s: str):
    try:
        if s is None or pd.isna(s): return None
    except Exception:
        if s is None: return None
    s = str(s).strip()
    if not s: return None
    m = re.search(r'\b([A-Z]{2})\s*,?\s*(\d{5})(?:-\d{4})?\b', s)
    if m and m.group(1).upper() in _US_STATES:
        return m.group(2)
    ms = list(re.finditer(r'\b(\d{5})(?:-\d{4})?\b', s))
    return ms[-1].group(1) if ms else None

@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address(addr: str, key: str | None):
    addr = (addr or "").strip()
    if not addr: return None
    m = LATLNG_RE.match(addr)
    if m:
        lat = float(m.group(1)); lng = float(m.group(2))
        return {"lat": lat, "lng": lng, "formatted": f"{lat:.6f}, {lng:.6f}", "source": "coord"}
    zip5 = _smart_zip_from_text(addr)
    if zip5:
        try:
            import pgeocode
            nomi = pgeocode.Nominatim("us")
            info = nomi.query_postal_code(zip5)
            if pd.notna(info.latitude) and pd.notna(info.longitude):
                return {"lat": float(info.latitude), "lng": float(info.longitude), "formatted": f"ZIP {zip5}", "source": "zip"}
        except Exception:
            pass
    if key:
        try:
            r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                             params={"address": addr, "key": key, "language": "en"},
                             timeout=15)
            j = r.json()
            if j.get("status") == "OK" and j.get("results"):
                g = j["results"][0]; loc = g["geometry"]["location"]
                return {"lat": float(loc["lat"]), "lng": float(loc["lng"]),
                        "formatted": g.get("formatted_address", addr), "source": "google"}
        except Exception:
            pass
    headers = {"User-Agent": "tech-map/1.0 (contact: support@example.com)"}
    osm_params = {"q": addr, "format": "json", "limit": 1, "addressdetails": 1, "countrycodes": "us", "accept-language": "en"}
    for url, src in [("https://nominatim.openstreetmap.org/search", "osm"),
                     ("https://geocode.maps.co/search", "mapsco")]:
        try:
            r2 = requests.get(url, params=osm_params, headers=headers, timeout=20)
            arr = r2.json()
            if isinstance(arr, list) and arr:
                it = arr[0]
                return {"lat": float(it["lat"]), "lng": float(it["lon"]),
                        "formatted": it.get("display_name", addr), "source": src}
        except Exception:
            pass
    return None

def _on_cust_addr_change():
    addr = st.session_state.get("cust_quick_addr", "").strip()
    if not addr:
        st.session_state.pop("cust_quick_pin", None); return
    gq = geocode_address(addr, GOOGLE_PLACES_KEY)
    if not gq:
        st.session_state["_last_cust_geocode_msg"] = "Geocoding failed: try adding city/state or use coords like 34.0522,-118.2437."
        st.session_state.pop("cust_quick_pin", None); return
    st.session_state['cust_quick_pin'] = {"lat": gq["lat"], "lng": gq["lng"], "formatted": gq.get("formatted", addr), "source": gq.get("source","")}
    pad = 0.2
    st.session_state["_zoom_bounds"] = [[gq["lat"]-pad, gq["lng"]-pad],[gq["lat"]+pad, gq["lng"]+pad]]
    st.session_state["_last_cust_geocode_msg"] = f"Located: {gq.get('formatted', addr)} (source: {gq.get('source','geocoder')})"

# ======================
# Data cleaning/fill (Technicians)
# ======================
df = st.session_state.df.copy()
df.columns = [str(c).strip() for c in df.columns]

# 别名映射（含 Level）
alias_map = {}
for c in list(df.columns):
    lc = c.lower().strip()
    if lc in {"lat","latitude","纬度","y","y_coord","ycoordinate","lat_dd","latitudes","lattitude"}:
        alias_map[c] = "Latitude"
    if lc in {"lon","lng","long","longitude","经度","x","x_coord","xcoordinate","lon_dd","longitudes","longtitude"}:
        alias_map[c] = "Longitude"
    if lc in {"联系人","contact","联系人姓名","联络人","contact name","contact person"}:
        alias_map[c] = "Contact"
    if lc in {"邮箱","email","电子邮件","e-mail","email address"}:
        alias_map[c] = "Email"
    if lc in {"备注","note","说明","备注信息","notes","remark"}:
        alias_map[c] = "Note"
    if lc in {"电话","phone","手机号","手机","联系电话","phone number","tel","telephone"}:
        alias_map[c] = "Phone"
    if lc in {"level", "等级", "级别", "lv", "lvl", "tier", "rank", "工级", "层级"}:
        alias_map[c] = "Level"

if alias_map:
    df.rename(columns=alias_map, inplace=True)

for col in ["Contact","Email","Note","Phone"]:
    if col not in df.columns:
        df[col] = pd.NA

def _pick_col(cols):
    for c in cols:
        if c in df.columns: return c
    return None

lat_col = _pick_col(['Latitude', 'Lat', 'latitude', 'lat'])
lon_col = _pick_col(['Longitude', 'Lon', 'Lng', 'longitude', 'lon', 'lng'])

def _smart_zip_from_text_wrap(s):
    z = _smart_zip_from_text(s)
    try:
        if z is None or pd.isna(z) or str(z).strip() == "": return pd.NA
    except Exception:
        if z is None or str(z).strip() == "": return np.nan
    return z

# backfill lat/lng from ZIP if needed
if not lat_col or not lon_col:
    zip_candidates = ['ZIP','Zip','zip','ZipCode','ZIP Code','PostalCode','Postal Code','postcode','Postcode','邮编']
    zip_col0 = next((c for c in zip_candidates if c in df.columns), None)
    if zip_col0 is not None:
        df['ZIP'] = df[zip_col0]
    elif 'Address' in df.columns:
        df['ZIP'] = df['Address'].apply(_smart_zip_from_text_wrap)
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
                try: return float(str(x).strip().replace(',', '.'))
                except: return np.nan
            df['Latitude']  = pd.to_numeric(df.get('Latitude'), errors='coerce')
            df['Longitude'] = pd.to_numeric(df.get('Longitude'), errors='coerce')
            df['Latitude']  = df['Latitude'].where(df['Latitude'].notna(),  df['latitude'].map(_to_num))
            df['Longitude'] = df['Longitude'].where(df['Longitude'].notna(), df['longitude'].map(_to_num))
            df.drop(columns=['postal_code','latitude','longitude'], inplace=True, errors='ignore')
            lat_col, lon_col = 'Latitude', 'Longitude'
    except Exception as e:
        st.warning(f"ZIP backfill failed: {e}")

if not lat_col or not lon_col:
    st.error(f"Missing lat/lng columns. Expect {['Latitude','Lat','latitude','lat']} / {['Longitude','Lon','Lng','longitude','lon','lng']} or provide Address/ZIP. Current: {list(df.columns)}")
    st.stop()

def clean_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(',', '.'); s = re.sub(r'[^0-9\.\-]', '', s)
    try: return float(s)
    except: return np.nan

df['Latitude']  = df[lat_col].apply(clean_num)
df['Longitude'] = df[lon_col].apply(clean_num)

if 'Level' not in df.columns:
    guess = next((c for c in df.columns if re.search(r'(?i)level|等级|级别|lv|lvl|tier|rank', str(c))), None)
    if guess and guess != 'Level':
        df.rename(columns={guess: 'Level'}, inplace=True)
if 'Level' not in df.columns:
    st.warning("未找到 'Level' 列，已默认所有技师 Level=3。")
    df['Level'] = 3

def to_level(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+)', str(x))
    if not m: return np.nan
    v = int(m.group(1)); return v if 1 <= v <= 7 else np.nan

df['Level'] = pd.to_numeric(df['Level'].apply(to_level), errors='coerce').astype('Int64')
df.loc[~df['Level'].between(1, 7), 'Level'] = pd.NA
df = df[(~df['Level'].eq(7)) | (df['Level'].isna())].copy()

for need in ['Name', 'Address']:
    if need not in df.columns:
        st.error(f"Missing required column: {need}"); st.stop()

zip_candidates = ['ZIP','Zip','zip','ZipCode','ZIP Code','PostalCode','Postal Code','postcode','Postcode','邮编']
zip_col = next((c for c in zip_candidates if c in df.columns), None)
df['ZIP']  = (df[zip_col] if zip_col is not None else df['Address'].apply(_smart_zip_from_text_wrap)).astype('string')
df['ZIP5'] = df['ZIP'].str.zfill(5)

try:
    import pgeocode
except ImportError:
    st.error("Missing dependency pgeocode: run  `python -m pip install pgeocode`"); st.stop()

nomi = pgeocode.Nominatim('us')
zip_list = df['ZIP5'].dropna().unique().tolist()
zip_ref = nomi.query_postal_code(zip_list)
zip_ref = zip_ref[['postal_code','latitude','longitude','state_code','place_name','county_name']].dropna(subset=['latitude','longitude'])
zip_ref['postal_code'] = zip_ref['postal_code'].astype(str).str.zfill(5)
df = df.merge(zip_ref, left_on='ZIP5', right_on='postal_code', how='left')

def combine_first_series(a, b):
    a = pd.Series(a); b = pd.Series(b)
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

# ===== 冷/热 标识修复 =====
def _to_bool_flag(x):
    try:
        if x is None or pd.isna(x): return False
    except Exception:
        if x is None: return False
    s = str(x).strip().lower()
    return s in {'是','yes','y','true','t','1','✓','✔','√','✅','hot','cold','heat','cool','ac','heating','both','冷热','制冷','供暖'}

def _normalize_name(s):
    return re.sub(r'[^a-z]', '', str(s).strip().lower())

COLD_COL_HINTS = {'iscold','cold','cool','ac','hvaccold','is_cold','冷','制冷','空调','冷气'}
HOT_COL_HINTS  = {'ishot','hot','heat','heating','hvachot','is_hot','热','供暖','暖气'}

if 'IsColdFlag' not in df.columns: df['IsColdFlag'] = False
if 'IsHotFlag'  not in df.columns: df['IsHotFlag']  = False

for c in df.columns:
    norm = _normalize_name(c)
    if norm in COLD_COL_HINTS:
        df['IsColdFlag'] = df['IsColdFlag'] | df[c].apply(_to_bool_flag)
    if norm in HOT_COL_HINTS:
        df['IsHotFlag']  = df['IsHotFlag']  | df[c].apply(_to_bool_flag)

st.session_state.df = df

zip_all = load_zip_all_cached()
cities_master, counties_master = build_city_county_master(zip_all)

# ======================
# 客户信息清洗/补点
# ======================
cust_df = st.session_state.get("cust_df", None)
if isinstance(cust_df, pd.DataFrame) and len(cust_df) > 0:
    cust_df = cust_df.copy()
    cust_df.columns = [str(c).strip() for c in cust_df.columns]
    c_alias = {}
    for c in list(cust_df.columns):
        lc = c.lower()
        if lc in {"name","customer","customername","客户","客户名"}: c_alias[c] = "CName"
        if lc in {"address","addr","地址"}: c_alias[c] = "CAddress"
        if lc in {"latitude","lat","纬度"}: c_alias[c] = "CLatitude"
        if lc in {"longitude","lon","lng","经度"}: c_alias[c] = "CLongitude"
        if lc in {"contact","联系人","联络人"}: c_alias[c] = "CContact"
    if c_alias:
        cust_df.rename(columns=c_alias, inplace=True)
    for col in ["CName","CAddress","CContact","CLatitude","CLongitude"]:
        if col not in cust_df.columns:
            cust_df[col] = pd.NA

    def _cnum(x):
        try:
            s = str(x).strip().replace(',', '.')
            return float(re.sub(r'[^0-9\.\-]','',s))
        except:
            return np.nan
    cust_df["CLatitude"]  = pd.to_numeric(cust_df["CLatitude"], errors='coerce').map(_cnum)
    cust_df["CLongitude"] = pd.to_numeric(cust_df["CLongitude"], errors='coerce').map(_cnum)

    def _zip_from_addr(a):
        try: a = str(a)
        except: return None
        return _smart_zip_from_text(a)

    need = cust_df["CLatitude"].isna() | cust_df["CLongitude"].isna()
    cust_df["CZIP"] = cust_df["CAddress"].apply(_zip_from_addr).astype('string')

    zlist = cust_df["CZIP"].dropna().unique().tolist()
    if zlist:
        zref = nomi.query_postal_code(zlist)[["postal_code","latitude","longitude","state_code"]].dropna(subset=["latitude","longitude"])
        zref["postal_code"] = zref["postal_code"].astype(str).str.zfill(5)
        cust_df = cust_df.merge(zref, left_on="CZIP", right_on="postal_code", how="left")
        cust_df["CLatitude"]  = cust_df["CLatitude"].where(cust_df["CLatitude"].notna(), cust_df["latitude"])
        cust_df["CLongitude"] = cust_df["CLongitude"].where(cust_df["CLongitude"].notna(), cust_df["longitude"])
        cust_df["CState"] = cust_df.get("CState", pd.Series(pd.NA, index=cust_df.index))
        cust_df["CState"] = cust_df["CState"].where(cust_df["CState"].notna(), cust_df["state_code"])
        cust_df.drop(columns=["postal_code","latitude","longitude","state_code"], inplace=True, errors="ignore")
    else:
        if "CState" not in cust_df.columns:
            cust_df["CState"] = pd.NA

    st.session_state.cust_df = cust_df
# ======================
# Sidebar filters + PERF 控件（第一块）
# ======================
with st.sidebar:
    geo_level = st.selectbox("Scope", ["County", "City"], index=0)

    states_for_level = sorted((counties_master if geo_level.startswith("County") else cities_master)['State'].unique().tolist())
    state_choice = st.selectbox('State', ['All'] + states_for_level)

    if geo_level.startswith("County"):
        units = sorted(counties_master.loc[counties_master['State']==state_choice, 'County'].unique().tolist()) if state_choice!='All' \
                else sorted(counties_master['County'].unique().tolist())[:5000]
        unit_label = "County"
    else:
        units = sorted(cities_master.loc[cities_master['State']==state_choice, 'City'].unique().tolist()) if state_choice!='All' \
                else sorted(cities_master['City'].unique().tolist())[:5000]
        unit_label = "City"
    unit_choice = st.selectbox(unit_label, ['All'] + units)

    level_filter_display = st.multiselect("Level filter (map)", [1,2,3,4,5,6], default=[1,2,3,4,5,6],
                                          help="只显示所选等级（支持单选/多选）")

    st.subheader("Scoring Rules (for rings)")
    radius_miles  = st.slider("Radius (mi)", 5, 50, 20, 5)
    min_good      = st.number_input("Good techs ≥ within radius", 1, 10, 2, 1)
    only_show_units = st.checkbox("Only show qualified areas", value=True)

    st.markdown("---")
    perf_mode = st.selectbox(
        "Performance",
        ["Clustered markers (default)", "Fast dots (all techs)", "Standard markers (no cluster)"],
        index=2,
        help="Fast dots 显示全部维修工为小圆点，INHOUSE 用大图钉。"
    )
    show_all_techs = st.checkbox("Show all technicians (ignore filters)", value=(perf_mode=="Fast dots (all techs)"))

# ========= 这里初始化 / 重置 max_tech_markers_val（不渲染控件） =========
if "max_tech_markers_val" not in st.session_state:
    st.session_state.max_tech_markers_val = DEFAULT_MAX_TECH_MARKERS
if st.session_state.get("last_state_for_max_tech") != state_choice:
    st.session_state["last_state_for_max_tech"] = state_choice
    st.session_state.max_tech_markers_val = 4000 if state_choice != "All" else DEFAULT_MAX_TECH_MARKERS

# ======================
# Sidebar PERF 控件（第二块，真正渲染）
# ======================
with st.sidebar:
    st.markdown("**Performance Limits**")
    cust_upper_cap = 60000 if state_choice != 'All' else 10000
    cust_default   = 60000 if state_choice != 'All' else DEFAULT_MAX_CUST_MARKERS
    max_tech_markers = st.number_input(
        "Max technician markers to render",
        min_value=500, max_value=10000, value=int(st.session_state.max_tech_markers_val), step=100,
        key="max_tech_markers_val",
        help="超出将自适应抽稀"
    )

    max_cust_markers = st.number_input(
        "Max customer markers to render (state view supports up to 30k)",
        min_value=200, max_value=cust_upper_cap, value=cust_default, step=100,
        help="选中具体州时可放大到 30,000"
    )

    near_pin_boost   = st.slider("Keep more markers near 📍 (mi)", 20, 200, DEFAULT_NEAR_PIN_BOOST_MI, 10)

    st.markdown("---")
    only_dup = st.checkbox("Only duplicate addresses (same addr ≥2)", value=False, key="only_dup_addr_v3")
    dup_export_slot = st.empty()
    st.session_state["_dup_export_slot"] = dup_export_slot

with st.sidebar:
    render_mode = st.radio(
        "🗺️ Map render mode",
        ["Static (HTML)", "Interactive (st_folium)"],
        index=0,
        key="render_mode_radio",
        help="If map not visible or need interactions, switch this."
    )
USE_STATIC_MAP = (render_mode == "Static (HTML)")

# ======================
# Area statistics（环形统计）
# ======================
base_mask = pd.Series(True, index=df.index)
if state_choice != 'All':
    base_mask &= (df['State'] == state_choice)
filtered_base = df.loc[base_mask].copy()

if geo_level.startswith("County"):
    base_master = counties_master if state_choice=='All' else counties_master[counties_master['State']==state_choice]
    name_col = 'County'; layer_name = "County Rings"
else:
    base_master = cities_master if state_choice=='All' else cities_master[cities_master['State']==state_choice]
    name_col = 'City'; layer_name = "City Rings"

if 'unit_choice' in locals() and unit_choice != 'All':
    base_master = base_master[base_master[name_col] == unit_choice]
base_master = base_master.copy()

points_all  = filtered_base.dropna(subset=['Latitude','Longitude']).copy()
points_good = points_all.copy()

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
            a = np.sin((latp-clat)/2)**2 + np.cos(clat)*np.cos(latp)*np.sin((lonp-clng)/2)**2
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
# Export stats (上方 metrics)
# ======================
unit_key = 'County' if geo_level.startswith("County") else 'City'
label_total = "Total counties" if unit_key == 'County' else "Total cities"
label_yes   = "Counties with techs" if unit_key == 'County' else "Cities with techs"
label_no    = "Counties without techs" if unit_key == 'County' else "Cities without techs"

cm_units = base_master.copy()
total_units  = len(cm_units)
covered_units = int(cm_units['meets'].sum())
empty_units   = total_units - covered_units
empty_rate    = (empty_units / total_units) if total_units else 0

left, mid, right, dl = st.columns([0.9, 0.9, 0.9, 1.1])
with left:  st.metric(label_total, f"{total_units:,}")
with mid:   st.metric(label_yes,   f"{covered_units:,}")
with right: st.metric(label_no, f"{empty_units:,} ({empty_rate:.1%} empty rate)")

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
        "Download",
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
        st.toast(f"Saved to: {save_path}")
    except Exception as e:
        st.warning(f"Could not save to local folder: {e}")

# ======================
# Search row
# ======================
c1, c2, c3 = st.columns([0.28, 0.36, 0.36])
with c1:
    q_name = st.text_input("Tech name", key="q_name", placeholder="e.g., ACME Tech", autocomplete="off")
with c2:
    q_addr = st.text_input("Tech address", key="q_addr", placeholder="City/State/Street/ZIP", autocomplete="off")
with c3:
    st.text_input("Customer address (press Enter to pin 📍)", key="cust_quick_addr",
                  placeholder="Press Enter after input / or coordinates 34.0522,-118.2437",
                  on_change=_on_cust_addr_change)

if "_last_cust_geocode_msg" in st.session_state:
    st.toast(st.session_state.pop("_last_cust_geocode_msg"))

# ======================
# Filter + matched set（含抽稀）
# ======================
def _s(val):
    try:
        if val is None: return ""
        if isinstance(val, float) and np.isnan(val): return ""
        if pd.isna(val): return ""
    except Exception:
        pass
    return str(val)

def _full_address_from_row(row):
    def _clean(v):
        try:
            if v is None or pd.isna(v): return ""
        except Exception:
            if v is None: return ""
        return str(v).strip()
    addr = _clean(row.get('Address', ''))
    if addr: return addr
    city  = _clean(row.get('City', ''))
    state = _clean(row.get('State', ''))
    zip5  = _clean(row.get('ZIP', ''))
    parts = [p for p in (city, state, zip5) if p]
    return ", ".join(parts) if parts else ""

def _norm_addr_for_dup(addr: str) -> str:
    if addr is None: return ""
    s = str(addr).strip().lower()
    s = s.replace('，', ',').replace(',', ' ')
    s = re.sub(r'[^a-z0-9]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

mask = pd.Series(True, index=df.index)
if state_choice != 'All':
    mask &= (df['State'] == state_choice)
if geo_level.startswith("County") and unit_choice != 'All':
    mask &= (df['County'] == unit_choice)
if (not geo_level.startswith("County")) and unit_choice != 'All':
    mask &= (df['City'] == unit_choice)

filtered = df.loc[mask].copy()
points = filtered.dropna(subset=['Latitude','Longitude']).copy()

_selected_display_lvls = [int(x) for x in (level_filter_display or [])]
if _selected_display_lvls:
    points = points[points['Level'].isin(_selected_display_lvls)].copy()
else:
    points = points.iloc[0:0].copy()

if show_all_techs or (perf_mode == "Fast dots (all techs)"):
    points = df.dropna(subset=['Latitude','Longitude']).copy()
    if _selected_display_lvls:
        points = points[points['Level'].isin(_selected_display_lvls)].copy()

dup_points_export = None
only_dup = st.session_state.get("only_dup_addr_v3", False)  # 兜底，保证有值
# —— 当勾选 Only duplicate 时，强制使用聚合模式（仅对本次渲染生效）—— #
if only_dup:
    perf_mode = "Clustered markers (default)"
    # 避免与 “Fast dots (all techs)” 冲突：重复模式不需要全量小圆点
    show_all_techs = False
if only_dup:
    addr_series_all = points.apply(_full_address_from_row, axis=1).map(_norm_addr_for_dup)
    vc = addr_series_all.value_counts()
    dup_mask = addr_series_all.map(vc).fillna(0).astype(int) >= 2
    dup_points_export = points[dup_mask].copy()
    points = dup_points_export.copy()
    

def _contains_safe(s, q):
    return s.astype(str).str.contains(re.escape(q), case=False, na=False)

matched = points.copy()
has_query = False
if st.session_state.get("q_name"):
    has_query = True
    matched = matched[_contains_safe(matched['Name'], st.session_state["q_name"])]
if st.session_state.get("q_addr"):
    has_query = True
    addr_q = st.session_state["q_addr"]
    addr_mask = (
        _contains_safe(points['Address'], addr_q) |
        _contains_safe(points.get('City',   pd.Series("", index=points.index)),   addr_q) |
        _contains_safe(points.get('County', pd.Series("", index=points.index)),   addr_q) |
        _contains_safe(points.get('ZIP',    pd.Series("", index=points.index)),   addr_q) |
        _contains_safe(points.get('State',  pd.Series("", index=points.index)),   addr_q)
    )
    matched = matched[addr_mask]
search_active = bool(has_query) and (len(matched) > 0)

def _is_inhouse(name: str) -> bool:
    return "INHOUSE-TECH" in str(name).upper()

inhouse_mask = points['Name'].astype(str).map(_is_inhouse)

matched_idx = None
if search_active:
    matched_idx = matched.index

cust_pin = st.session_state.get("cust_quick_pin")

# 这里使用 number_input 的值（总是存在）
max_tech_markers = int(st.session_state.get("max_tech_markers_val", DEFAULT_MAX_TECH_MARKERS))

idx_kept = thin_points(
    points, target_max=max_tech_markers,
    matched_idx=matched_idx,
    inhouse_mask=inhouse_mask,
    cust_pin=cust_pin,
    near_radius_mi=near_pin_boost,
)
points = points.loc[idx_kept].copy()
if search_active:
    matched = matched.loc[matched.index.intersection(points.index)]

# 客户点：先按州过滤，再按上限抽样
cust_df = st.session_state.get("cust_df", None)
cust_for_map = None
if isinstance(cust_df, pd.DataFrame) and len(cust_df) > 0:
    cand = cust_df.dropna(subset=["CLatitude","CLongitude"]).copy()
    if state_choice != 'All' and "CState" in cand.columns:
        cand = cand[cand["CState"].astype("string") == str(state_choice)]

    if len(cand) > max_cust_markers:
        if cust_pin:
            lat = cand["CLatitude"].to_numpy(float)
            lng = cand["CLongitude"].to_numpy(float)
            d = _haversine_batch(cust_pin["lat"], cust_pin["lng"], lat, lng)
            near = cand.iloc[np.where(d <= near_pin_boost)[0]]
            if len(near) >= max_cust_markers:
                cust_for_map = near.sample(max_cust_markers, random_state=1)
            else:
                remain_quota = max_cust_markers - len(near)
                others = cand.drop(index=near.index)
                cust_for_map = pd.concat([near, others.sample(remain_quota, random_state=1)])
        else:
            cust_for_map = cand.sample(max_cust_markers, random_state=1)
    else:
        cust_for_map = cand
else:
    cust_for_map = None

# ======================
# Map drawing
# ======================
US_STATES_GEO_PATH = os.path.join(data_dir, "us_states.geojson")
LEVEL_COLORS = {1:'#2ecc71',2:'#FFD700',3:'#FF4D4F',4:'#FFC0CB',5:'#8A2BE2',6:'#000000'}
prefer_canvas = True

m = folium.Map(location=[37.8, -96.0], zoom_start=4, keyboard=False, prefer_canvas=prefer_canvas, tiles=None)
folium.TileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", attr="&copy; CARTO", name="Carto Positron", control=True, max_zoom=20, overlay=False).add_to(m)
folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", attr="&copy; OpenStreetMap contributors", name="OSM", control=True, max_zoom=19, overlay=False).add_to(m)
m.get_root().header.add_child(folium.Element("""
<style>
.leaflet-container:focus, .leaflet-container:focus-visible { outline: none !important; }
.leaflet-interactive:focus { outline: none !important; }
</style>
"""))

# 州边界（点量大时跳过）
states_geo = None
if len(points) <= 6000:
    states_geo = load_us_states_geojson_cached(US_STATES_GEO_PATH)

selected_bounds = None
if states_geo:
    feats = states_geo['features']
    def style_fn(feat):
        code = feat.get('id') or feat.get('properties', {}).get('state_code')
        is_selected = (state_choice != 'All' and code == state_choice)
        return {'fillColor':'#ffffff','fillOpacity':0.0,'color':'#2563eb','weight':3.0 if is_selected else 1.8}
    gj = folium.GeoJson(states_geo, name="US States", style_function=style_fn,
                        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'])).add_to(m)
    if state_choice != 'All':
        target = next((f for f in feats if (f.get('id') or f.get('properties', {}).get('state_code')) == state_choice), None)
        if target:
            def iter_coords(geom):
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

# County/City rings
radius_m = radius_miles * 1609.34
unit_fg = folium.FeatureGroup(name=("County Rings" if geo_level.startswith("County") else "City Rings"), show=True).add_to(m)
for _, r in centroids_to_plot.iterrows():
    ring_color = '#1e88e5' if bool(r['meets']) else '#9e9e9e'
    tip = (f"{'County' if geo_level.startswith('County') else 'City'}: {r.get('County' if geo_level.startswith('County') else 'City')} "
           f"({r.get('State')}) | Good: {int(r['good_in_radius'])} / All: {int(r['all_in_radius'])}")
    folium.Circle([r['cLat'], r['cLng']], radius=radius_m,
                  color=ring_color, weight=1.6, fill=True, fill_opacity=0.05,
                  tooltip=tip).add_to(unit_fg)

# 弹窗
POPUP_MAX_W = 520
def make_worker_popup(name, level, address=None, distance_text="", contact=None, email=None, phone=None, note=None,
                      is_cold=False, is_hot=False, **kwargs):
    def _has(v):
        try:
            if v is None or pd.isna(v): return False
        except Exception:
            if v is None: return False
        return str(v).strip() != ""
    icons_html = ""
    if bool(is_cold): icons_html += "<span style='font-size:18px;line-height:1'>🔧</span>"
    if bool(is_hot):  icons_html += "<span style='font-size:18px;line-height:1'>🔥</span>"
    contact_html = f"<div><b>Contact:</b> {str(contact)}</div>" if _has(contact) else ""
    phone_html   = f"<div><b>Phone:</b> {str(phone)}</div>"     if _has(phone)   else ""
    email_html   = f'<div><b>Email:</b> <a href="mailto:{str(email)}" target="_blank">{str(email)}</a></div>' if _has(email) else ""
    note_html    = f"<div><b>Note:</b> {str(note)}</div>" if _has(note) else ""
    html = f"""
    <div style="min-width:420px; font-size:13px; line-height:1.4; white-space:normal;">
      <div><b>Name:</b> {str(name)} {icons_html}</div>
      <div><b>Level:</b> {str(level)}</div>
      <div><b>Address:</b> {str(address or '')}</div>
      {contact_html}{phone_html}{email_html}{note_html}
      <div><b>Distance:</b> {str(distance_text)}</div>
    </div>
    """
    return folium.Popup(html, max_width=POPUP_MAX_W)

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
def popup_distance_text(lat, lng, prefer_drive=False):
    if not cust_pin: return "-"
    dline = haversine_miles(cust_pin["lat"], cust_pin["lng"], float(lat), float(lng))
    if prefer_drive:
        drive = osrm_drive_info(cust_pin["lat"], cust_pin["lng"], float(lat), float(lng))
        if drive:
            return f"{drive[0]:.1f} mi · {int(round(drive[1]))} min"
    return f"{dline:.1f} mi (straight-line)"

def _is_inhouse(name: str) -> bool:
    return "INHOUSE-TECH" in str(name).upper()

def _make_marker_icon(color_hex: str, larger: bool = False):
    return ring_pin_icon(color_hex, size_h_px=(54 if larger else 28))

# 拓点避免重叠（技师）
points = points.copy()
points['LatAdj'] = points['Latitude'].values
points['LngAdj'] = points['Longitude'].values
if not points.empty:
    grp = points.groupby(['Latitude','Longitude'])
    for (lat0, lng0), idxs in grp.groups.items():
        sub = points.loc[idxs]
        if sub['Name'].astype(str).nunique() > 1 and len(sub) > 1:
            k = len(sub); delta = 0.00035
            lat_rad = np.radians(lat0 if pd.notna(lat0) else 0.0)
            for j, idx in enumerate(sub.index):
                ang  = 2*np.pi * (j / k)
                dlat = delta * np.cos(ang)
                dlng = (delta / max(0.15, np.cos(lat_rad))) * np.sin(ang)
                points.at[idx, 'LatAdj'] = float(lat0) + dlat
                points.at[idx, 'LngAdj'] = float(lng0) + dlng

use_cluster = (perf_mode == "Clustered markers (default)")
force_all = show_all_techs or (perf_mode == "Fast dots (all techs)")
always_big_inhouse = force_all

workers_fg = folium.FeatureGroup(name="Technician points", show=True).add_to(m)

disable_popups = should_disable_popups(len(points))

def _lite_popup_row(row):
    addr = (row.get('Address') or f"{row.get('City','')}, {row.get('State','')}, {row.get('ZIP','')}")
    dist = popup_distance_text(row['LatAdj'], row['LngAdj'], prefer_drive=False)
    icons = f"{'🔧' if bool(row.get('IsColdFlag',False)) else ''}{'🔥' if bool(row.get('IsHotFlag',False)) else ''}"
    html = f"""
    <div style="min-width:240px; font-size:12.5px; line-height:1.35; white-space:normal;">
      <div><b>Name:</b> {str(row.get('Name',''))} {icons}</div>
      <div><b>Level:</b> {str(row.get('Level',''))}</div>
      <div><b>Address:</b> {str(addr)}</div>
      <div><b>Distance:</b> {str(dist)}</div>
    </div>
    """
    return folium.Popup(html, max_width=320)

if use_cluster:
    clusters = {}
    present_levels = sorted(points['Level'].dropna().astype(int).unique().tolist()) if not points.empty else []
    for lvl in present_levels:
        col = LEVEL_COLORS.get(int(lvl), '#3388ff')
        clusters[int(lvl)] = MarkerCluster(
            name=f"Level {int(lvl)}",
            icon_create_function=f"""
            function(cluster) {{
              var count = cluster.getChildCount();
              return new L.DivIcon({{
                html: '<div style="background:{col};opacity:0.85;border-radius:20px;width:32px;height:32px;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;border:2px solid white;font-size:12px;">'+count+'</div>',
                className: 'marker-cluster', iconSize: new L.Point(32, 32)
              }});
            }}
            """
        ).add_to(m)

    others_fg = folium.FeatureGroup(name="Others (no level)", show=True).add_to(m)

    for _, row in points.iterrows():
        lvl = int(row['Level']) if not pd.isna(row['Level']) else None
        base_color = LEVEL_COLORS.get(lvl, '#3388ff')
        larger = always_big_inhouse and _is_inhouse(row.get('Name',''))
        icon = _make_marker_icon('#1E90FF' if larger else base_color, larger=larger)
        if disable_popups:
            popup_obj = None
        else:
            distance_text = popup_distance_text(row['LatAdj'], row['LngAdj'], prefer_drive=False)
            popup_obj = make_worker_popup(
                name=row.get('Name',''), level=row.get('Level',''),
                address=(row.get('Address') or f"{row.get('City','')}, {row.get('State','')}, {row.get('ZIP','')}"),
                distance_text=distance_text,
                contact=row.get('Contact'), email=row.get('Email'), phone=row.get('Phone'), note=row.get('Note'),
                is_cold=row.get('IsColdFlag', False), is_hot=row.get('IsHotFlag',  False),
            )
        target_layer = clusters.get(lvl, None) or others_fg
        folium.Marker([row['LatAdj'], row['LngAdj']], icon=icon, popup=popup_obj,
                      tooltip=str(row.get('Name',''))).add_to(target_layer)
else:
    is_fast_dots = (perf_mode == "Fast dots (all techs)")
    dot_r = 7
    present_levels_nc = sorted(points['Level'].dropna().astype(int).unique().tolist()) if not points.empty else []
    lvl_groups_nc = {int(lvl): folium.FeatureGroup(name=f"Level {int(lvl)}", show=True).add_to(m) for lvl in present_levels_nc}
    others_fg_nc = folium.FeatureGroup(name="Others (no level)", show=True).add_to(m)

    def _add_circle(row, base_color, target_layer, radius=None):
        folium.CircleMarker(
            [row['LatAdj'], row['LngAdj']],
            radius=(radius or dot_r),
            stroke=False, weight=0,
            fill=True, fill_color=base_color, fill_opacity=0.8,
            tooltip=str(row.get('Name','')),
            popup=None if disable_popups else _lite_popup_row(row)
        ).add_to(target_layer)

    def _add_inhouse_marker(row, target_layer):
        icon = _make_marker_icon('#1E90FF', larger=True)
        folium.Marker([row['LatAdj'], row['LngAdj']], icon=icon,
                      tooltip=str(row.get('Name','')),
                      popup=None if disable_popups else _lite_popup_row(row)).add_to(target_layer)

    for _, row in points.iterrows():
        lvl = int(row['Level']) if not pd.isna(row['Level']) else None
        base_color = LEVEL_COLORS.get(lvl, '#3388ff')
        target_layer = lvl_groups_nc.get(lvl, others_fg_nc)
        if _is_inhouse(row.get('Name','')) and (always_big_inhouse or not is_fast_dots):
            _add_inhouse_marker(row, target_layer)
        else:
            if is_fast_dots:
                _add_circle(row, base_color, target_layer, radius=dot_r)
            else:
                icon = _make_marker_icon(base_color, larger=False)
                folium.Marker([row['LatAdj'], row['LngAdj']], icon=icon,
                              tooltip=str(row.get('Name','')),
                              popup=None if disable_popups else _lite_popup_row(row)).add_to(target_layer)

# 搜索命中红旗
def render_hit_flags(map_obj, matched_df):
    if matched_df is None or matched_df.empty: return
    for _, r in matched_df.iterrows():
        dist_txt = popup_distance_text(r['Latitude'], r['Longitude'], prefer_drive=True)
        popup_obj = make_worker_popup(
            name=r.get('Name',''), level=r.get('Level',''),
            address=(r.get('Address') or f"{r.get('City','')}, {r.get('State','')}, {r.get('ZIP','')}"),
            distance_text=dist_txt,
            contact=r.get('Contact'), email=r.get('Email'), phone=r.get('Phone'), note=r.get('Note'),
            is_cold=r.get('IsColdFlag', False), is_hot=r.get('IsHotFlag',  False),
        )
        folium.Marker([float(r['Latitude']), float(r['Longitude'])],
                      icon=big_flag_icon(size_px=42),
                      tooltip=f"🔎 Match: {str(r.get('Name',''))}",
                      popup=popup_obj,
                      z_index_offset=10000).add_to(map_obj)

if search_active:
    render_hit_flags(m, matched)

# 客户针脚（单次输入📍）
if cust_pin:
    p = cust_pin
    folium.Marker(
        [float(p["lat"]), float(p["lng"])],
        icon=customer_pin_icon(size_px=42),
        tooltip=f"📍 {p.get('formatted', 'Customer address')}",
        popup=f"<b>Customer:</b> {str(p.get('formatted'))}",
        z_index_offset=12000
    ).add_to(m)

# 客户上传🏠（州过滤+上限后） —— ==== CUST-STATE & 30K ====
cust_count_for_badge = 0
if isinstance(cust_for_map, pd.DataFrame) and (len(cust_for_map) > 0):
    cust_points = cust_for_map.copy()
    cust_count_for_badge = int(len(cust_points))

    # 3K+ 客户改用 CircleMarker 以保证流畅；关闭弹窗，仅保留 tooltip
    heavy_cust = len(cust_points) >= 5000
    cust_fg = folium.FeatureGroup(
        name=f"Customers (uploaded{'' if state_choice=='All' else ' - ' + str(state_choice)})",
        show=True
    ).add_to(m)

    def _cust_popup_row(row):
        name = row.get("CName","")
        addr = row.get("CAddress","")
        contact = row.get("CContact","")
        lat = row.get("CLatitude", np.nan)
        lng = row.get("CLongitude", np.nan)
        dist = "-"
        try:
            if pd.notna(lat) and pd.notna(lng):
                dist = popup_distance_text(float(lat), float(lng), prefer_drive=False)
        except Exception:
            pass
        html = f"""
        <div style="min-width:240px; font-size:12.5px; line-height:1.35;">
          <div><b>Name:</b> {str(name)}</div>
          <div><b>Address:</b> {str(addr)}</div>
          <div><b>Contact:</b> {str(contact) if pd.notna(contact) else ''}</div>
          <div><b>Distance:</b> {str(dist)}</div>
        </div>
        """
        return folium.Popup(html, max_width=320)

    if heavy_cust:
        # 轻量渲染（推荐 30k）
        for _, r in cust_points.iterrows():
            folium.CircleMarker(
                [float(r["CLatitude"]), float(r["CLongitude"])],
                radius=6, stroke=False, weight=0,
                fill=True, fill_opacity=0.7,
                tooltip=str(r.get("CName",""))
            ).add_to(cust_fg)
    else:
        # 轻量阈值以下仍可用小房子图标
        for _, r in cust_points.iterrows():
            icon = customer_house_icon(size_px=24)
            folium.Marker(
                [float(r["CLatitude"]), float(r["CLongitude"])],
                icon=icon,
                tooltip=str(r.get("CName","")),
                popup=None if should_disable_popups(len(points)) else _cust_popup_row(r),
                z_index_offset=9000
            ).add_to(cust_fg)

# 初始视野
CONUS_BOUNDS = [[24.5, -125.0], [49.5, -66.9]]
def fit_initial_or_search(map_obj, nat_bnds, state_bnds, matched_df, search_active):
    if "_zoom_bounds" in st.session_state:
        map_obj.fit_bounds(st.session_state.pop("_zoom_bounds")); return
    if search_active and matched_df is not None and len(matched_df) > 0:
        if len(matched_df) == 1:
            lat = float(matched_df['Latitude'].iloc[0]); lng = float(matched_df['Longitude'].iloc[0])
            pad = 0.35; b = [[lat - pad, lng - pad], [lat + pad, lng + pad]]
        else:
            b = [[matched_df['Latitude'].min(),  matched_df['Longitude'].min()],
                 [matched_df['Latitude'].max(),  matched_df['Longitude'].max()]]
        map_obj.fit_bounds(b); return
    if (state_choice != 'All') and state_bnds:
        map_obj.fit_bounds(state_bnds); return
    map_obj.fit_bounds(nat_bnds)

fit_initial_or_search(m, CONUS_BOUNDS, selected_bounds, matched if 'matched' in locals() else None, search_active)

# 图例（右上角）
lvl_order = [1,2,3,4,5,6]
lvl_counts = {lvl: int(points['Level'].eq(lvl).sum()) for lvl in lvl_order}
rows_html = "".join([
    f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
    f"<span style='display:inline-block;width:10px;height:10px;background:{LEVEL_COLORS[lvl]};border-radius:2px'></span>"
    f"<span>{lvl}</span>"
    f"<span style='margin-left:auto'>{lvl_counts.get(lvl,0)}</span>"
    f"</div>"
    for lvl in lvl_order
])
legend_html = f"""
<div style="
  position: fixed; top: 10px; right: 12px; z-index: 9999;
  background: #fff; padding: 6px 8px; border-radius: 8px;
  box-shadow:0 2px 6px rgba(0,0,0,.20); font-size: 12px; line-height: 1.4;
  width: max-content; max-width: 220px; border: 1px solid rgba(0,0,0,.08);
">
  <div style="font-weight:600; margin-bottom:6px;">Level colors</div>
  {rows_html}
  <div style='margin-top:6px; border-top:1px dashed #e5e7eb; padding-top:6px; font-weight:600;'>
    Total techs: {int(points.shape[0])}
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# —— 客户数量：地图左上角小字 —— #
cust_badge_html = f"""
<div style="
  position: fixed; top: 10px; left: 12px; z-index: 9999;
  background: rgba(255,255,255,.95); padding: 3px 6px; border-radius: 6px;
  border: 1px solid #e5e7eb; font-size: 11px; line-height:1.2;
">
  Customers (rendered): {cust_count_for_badge}
</div>
"""
m.get_root().html.add_child(folium.Element(cust_badge_html))

# Layer control
folium.LayerControl(collapsed=True, position='topleft').add_to(m)

with st.sidebar:
    st.markdown("---")
    st.caption(f"🔑 Google Places: {'✅ Loaded' if GOOGLE_PLACES_KEY else '❌ Not set'}  {_mask_key(GOOGLE_PLACES_KEY)}")

# Render map
def render_map_once(m):
    try:
        if USE_STATIC_MAP:
            from streamlit.components.v1 import html
            html_str = m._repr_html_()
            html(html_str, height=760, scrolling=False)
            st.caption("✅ Rendered: static HTML")
        else:
            map_height = 760
            st_folium(m, use_container_width=True, height=map_height)
            st.caption("✅ Rendered: interactive (st_folium)")
    except Exception as e:
        st.error(f"Map render failed: {e}")
        st.stop()

render_map_once(m)

# ======================
# “Only duplicate” 正下方的导出按钮（紧挨着）
# ======================
if only_dup:
    slot = st.session_state.get("_dup_export_slot", None)
    if slot and dup_points_export is not None and not dup_points_export.empty:
        addr_series_all = dup_points_export.apply(_full_address_from_row, axis=1).map(_norm_addr_for_dup)
        vc = addr_series_all.value_counts()
        dup_points_export2 = dup_points_export.copy()
        dup_points_export2.insert(0, "Duplicate Count", addr_series_all.map(vc).astype(int).values)
        cols_pref = ["Duplicate Count","Name","Address","City","State","ZIP","Level","Latitude","Longitude","County"]
        cols_exist = [c for c in cols_pref if c in dup_points_export2.columns]
        dup_buf = _build_xlsx(dup_points_export2[cols_exist], sheet_name="DuplicateAddresses")
        with st.sidebar:
            slot.download_button(
                "Export duplicate addresses (Excel)",
                data=dup_buf,
                file_name=f"duplicate_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_dup_excel_sidebar"
            )
    elif slot:
        with st.sidebar:
            slot.info("No duplicate addresses under current filters.")
