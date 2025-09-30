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

# ---------- Regex ----------
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

# ---------- Icons ----------
def ring_pin_icon(color_hex: str = "#1E90FF", size_h_px: int = 38):
    """
    color_hex: ring color
    size_h_px: icon height in px (width auto 0.75 ratio)
    """
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

# Back-compat alias
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

[data-testid="stMetricValue"] { font-size: 18px !important; line-height:1.1!important; font-weight:700!important; }
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
# Session init
# ======================
if "df" not in st.session_state:
    st.session_state.df = None
if "data_meta" not in st.session_state:
    st.session_state.data_meta = {}

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
        st.error(f"Failed to load {_files[0]}: {e}")

df = st.session_state.get("df", None)
if df is None:
    # Sidebar data source (when no data yet)
    with st.sidebar:
        st.markdown("---")
        with st.expander("📁 Data Source (fixed folder)", expanded=True):
            new_dir = st.text_input("Data folder path", value=st.session_state.data_dir_path)
            if new_dir != st.session_state.data_dir_path:
                st.session_state.data_dir_path = new_dir
            os.makedirs(st.session_state.data_dir_path, exist_ok=True)

            files2 = [f for f in os.listdir(st.session_state.data_dir_path) if f.lower().endswith(SUPPORT_EXTS)]
            files2 = sorted(files2, key=lambda f: os.path.getmtime(os.path.join(st.session_state.data_dir_path, f)), reverse=True)

            if files2:
                pick = st.selectbox("Pick saved data file", files2, index=0)
                if st.button("Load selected"):
                    try:
                        path = os.path.join(st.session_state.data_dir_path, pick)
                        st.session_state.df = _load_df(path)
                        st.session_state.data_meta = {
                            "filename": pick,
                            "path": path,
                            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.success(f"Loaded: {pick}")
                        _safe_rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")

            new_file = st.file_uploader("Upload new data (saved into folder)", type=['csv','xlsx','xls'])
            if new_file is not None:
                try:
                    saved_path = _save_uploaded(new_file, st.session_state.data_dir_path)
                    st.session_state.df = _load_df(saved_path)
                    st.session_state.data_meta = {
                        "filename": os.path.basename(saved_path),
                        "path": saved_path,
                        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success(f"Uploaded & loaded: {os.path.basename(saved_path)}")
                    _safe_rerun()
                except Exception as e:
                    st.error(f"Upload/read failed: {e}")

    st.warning("No dataset loaded. Use the left panel (**📁 Data Source**) to pick or upload a file.")
    st.stop()

# ======================
# Cached refs
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

# ---- Address helpers ----
LATLNG_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$')
_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
    "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
    "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR"
}

def _smart_zip_from_text(s: str):
    try:
        if s is None or pd.isna(s):
            return None
    except Exception:
        if s is None:
            return None
    s = str(s).strip()
    if not s:
        return None
    m = re.search(r'\b([A-Z]{2})\s*,?\s*(\d{5})(?:-\d{4})?\b', s)
    if m and m.group(1).upper() in _US_STATES:
        return m.group(2)
    ms = list(re.finditer(r'\b(\d{5})(?:-\d{4})?\b', s))
    return ms[-1].group(1) if ms else None

@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address(addr: str, key: str | None):
    addr = (addr or "").strip()
    if not addr:
        return None
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
                return {
                    "lat": float(info.latitude), "lng": float(info.longitude),
                    "formatted": f"ZIP {zip5}",
                    "source": "zip"
                }
        except Exception:
            pass
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
        st.session_state.pop("cust_quick_pin", None)
        return
    gq = geocode_address(addr, GOOGLE_PLACES_KEY)
    if not gq:
        st.session_state["_last_cust_geocode_msg"] = "Geocoding failed: try adding city/state or use coords like 34.0522,-118.2437."
        st.session_state.pop("cust_quick_pin", None)
        return
    st.session_state['cust_quick_pin'] = {"lat": gq["lat"], "lng": gq["lng"], "formatted": gq.get("formatted", addr), "source": gq.get("source","")}
    pad = 0.2
    st.session_state["_zoom_bounds"] = [[gq["lat"]-pad, gq["lng"]-pad],[gq["lat"]+pad, gq["lng"]+pad]]
    src = gq.get("source","geocoder")
    st.session_state["_last_cust_geocode_msg"] = f"Located: {gq.get('formatted', addr)} (source: {src})"

# ======================
# Data cleaning/fill
# ======================
df.columns = [str(c).strip() for c in df.columns]
alias_map = {}
for c in list(df.columns):
    lc = c.lower()
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
if alias_map:
    df.rename(columns=alias_map, inplace=True)

for col in ["Contact","Email","Note","Phone"]:
    if col not in df.columns:
        df[col] = pd.NA

def _pick_col(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None
lat_col = _pick_col(['Latitude', 'Lat', 'latitude', 'lat'])
lon_col = _pick_col(['Longitude', 'Lon', 'Lng', 'longitude', 'lon', 'lng'])

def _smart_zip_from_text_wrap(s):
    z = _smart_zip_from_text(s)
    try:
        if z is None or pd.isna(z) or str(z).strip() == "":
            return pd.NA
    except Exception:
        if z is None or str(z).strip() == "":
            return np.nan
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
        st.warning(f"ZIP backfill failed: {e}")

if not lat_col or not lon_col:
    st.error(f"Missing lat/lng columns. Expect {['Latitude','Lat','latitude','lat']} / {['Longitude','Lon','Lng','longitude','lon','lng']} or provide Address/ZIP. Current: {list(df.columns)}")
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
    st.error("Missing required column: Level")
    st.stop()

def to_level(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+)', str(x))
    if not m: return np.nan
    v = int(m.group(1))
    return v if 1 <= v <= 7 else np.nan

df['Level'] = pd.to_numeric(df['Level'].apply(to_level), errors='coerce').astype('Int64')
df.loc[~df['Level'].between(1, 7), 'Level'] = pd.NA

for need in ['Name', 'Address']:
    if need not in df.columns:
        st.error(f"Missing required column: {need}")
        st.stop()

zip_candidates = ['ZIP','Zip','zip','ZipCode','ZIP Code','PostalCode','Postal Code','postcode','Postcode','邮编']
zip_col = next((c for c in zip_candidates if c in df.columns), None)
if zip_col is not None:
    df['ZIP'] = df[zip_col]
else:
    df['ZIP'] = df['Address'].apply(_smart_zip_from_text_wrap)
df['ZIP']  = df['ZIP'].apply(lambda x: _smart_zip_from_text_wrap(x)).astype('string')
df['ZIP5'] = df['ZIP'].str.zfill(5)

try:
    import pgeocode
except ImportError:
    st.error("Missing dependency pgeocode: run  `python -m pip install pgeocode`")
    st.stop()

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

# === Cold/Hot boolean normalization ===
def _to_bool_cn_en(x):
    try:
        if x is None or pd.isna(x):
            return False
    except Exception:
        if x is None:
            return False
    s = str(x).strip().lower()

    TRUE_TOKENS  = {'是','yes','y','true','t','1','✓','✔','√','✅'}
    FALSE_TOKENS = {'否','no','n','false','f','0','×','x','✗','✕','❌','-',''}

    if s in TRUE_TOKENS:  return True
    if s in FALSE_TOKENS: return False
    try:
        return bool(int(s))
    except Exception:
        return False

if 'IsColdFlag' not in df.columns:
    df['IsColdFlag'] = df.get('Is Cold', pd.Series(False, index=df.index)).apply(_to_bool_cn_en)
else:
    df['IsColdFlag'] = df['IsColdFlag'].apply(_to_bool_cn_en)
if 'IsHotFlag' not in df.columns:
    df['IsHotFlag'] = df.get('Is Hot',  pd.Series(False, index=df.index)).apply(_to_bool_cn_en)
else:
    df['IsHotFlag'] = df['IsHotFlag'].apply(_to_bool_cn_en)

zip_all = load_zip_all_cached()
cities_master, counties_master = build_city_county_master(zip_all)

# ======================
# Sidebar filters
# ======================
with st.sidebar:
    geo_level = st.selectbox("Scope", ["County", "City"], index=0)

    levels_present = sorted([int(x) for x in df['Level'].dropna().unique().tolist()]) or [1,2,3,4,5,6,7]
    level_choice = st.selectbox('Level filter', ['All'] + levels_present, index=0)

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

    st.subheader("Scoring Rules")
    good_levels   = st.multiselect("Tech level", [1,2,3,4,5,6,7], default=[1,2,3,4,5,6])
    radius_miles  = st.slider("Radius (mi)", 5, 50, 20, 5)
    min_good      = st.number_input("Good techs ≥ within radius", 1, 10, 2, 1)
    only_show_units       = st.checkbox("Show only qualified areas", value=True)
    only_show_good_points = st.checkbox("Show only good techs", value=True)
    st.checkbox("Only duplicate addresses (same addr ≥2)", value=False, key="only_dup_addr_v3")
    dup_export_slot = st.empty()

    st.markdown("---")
    source_mode = st.radio("Online enrichment source", ["Auto (Google first)", "Google only (faster)", "OSM only (fallback)"], index=0)
    if "hvac_only" not in st.session_state:
        st.session_state.hvac_only = False
    hvac_only = st.checkbox("HVAC-only companies", value=st.session_state.hvac_only, key="hvac_only")
    st.checkbox("Only show web-added (Level=7)", value=False, key="show_only_new")

    st.markdown("---")
    with st.expander("⚡ Performance", expanded=False):
        st.checkbox("Marker clustering (faster for many points)", key="perf_use_cluster", value=st.session_state.get("perf_use_cluster", True))
        st.checkbox("Canvas vector render", key="perf_prefer_canvas", value=st.session_state.get("perf_prefer_canvas", True))
        st.slider("Max rendered areas (county/city circles)", 200, 5000, int(st.session_state.get("perf_max_units", 1500)), 100, key="perf_max_units")
        st.checkbox("Ultra-fast (dots instead of pins)", key="perf_fast_dots",
            value=st.session_state.get("perf_fast_dots", True))
        st.slider("Fast-mode threshold (#points)", 500, 20000,
          int(st.session_state.get("perf_fast_threshold", 2500)), 100,
          key="perf_fast_threshold")
        st.slider("Dot radius (px)", 2, 12,
          int(st.session_state.get("perf_fast_radius", 8)), 1,
          key="perf_fast_radius")

with st.sidebar:
    st.markdown("---")
    with st.expander("📁 Data Source (fixed folder)", expanded=False):
        new_dir = st.text_input("Data folder path", value=st.session_state.data_dir_path)
        if new_dir != st.session_state.data_dir_path:
            st.session_state.data_dir_path = new_dir
        os.makedirs(st.session_state.data_dir_path, exist_ok=True)

        files2 = [f for f in os.listdir(st.session_state.data_dir_path) if f.lower().endswith(SUPPORT_EXTS)]
        files2 = sorted(files2, key=lambda f: os.path.getmtime(os.path.join(st.session_state.data_dir_path, f)), reverse=True)

        if files2:
            pick = st.selectbox("Pick saved data file", files2, index=0, key="pick_file_bottom")
            if st.button("Load selected", key="btn_load_selected_bottom"):
                try:
                    path = os.path.join(st.session_state.data_dir_path, pick)
                    st.session_state.df = _load_df(path)
                    st.session_state.data_meta = {"filename": pick, "path": path, "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    st.success(f"Loaded: {pick}")
                    _safe_rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")
        else:
            st.info("No data files (csv/xlsx/xls) in this folder.")

        new_file = st.file_uploader("Upload new data (save into folder)", type=['csv', 'xlsx', 'xls'], key="uploader_new_bottom")
        if new_file is not None:
            try:
                saved_path = _save_uploaded(new_file, st.session_state.data_dir_path)
                st.session_state.df = _load_df(saved_path)
                st.session_state.data_meta = {"filename": os.path.basename(saved_path), "path": saved_path, "loaded_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                st.success(f"Uploaded & loaded: {os.path.basename(saved_path)}")
                _safe_rerun()
            except Exception as e:
                st.error(f"Upload/read failed: {e}")

        if st.session_state.get("df") is not None:
            meta = st.session_state.get("data_meta", {})
            st.success(
                f"**{meta.get('filename','(untitled)')}**\n\n"
                f"Path: {meta.get('path','')}\n\n"
                f"Loaded at: {meta.get('loaded_at','')}\n\n"
                f"Rows: {len(st.session_state.df)}"
            )

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
# Area statistics
# ======================
base_mask = pd.Series(True, index=df.index)
if level_choice != 'All':
    base_mask &= (df['Level'] == level_choice)
if state_choice != 'All':
    base_mask &= (df['State'] == state_choice)
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

if geo_level.startswith("County"):
    base_master = counties_master if state_choice=='All' else counties_master[counties_master['State']==state_choice]
    name_col = 'County'; layer_name = "County Rings"
else:
    base_master = cities_master if state_choice=='All' else cities_master[cities_master['State']==state_choice]
    name_col = 'City'; layer_name = "City Rings"
if 'unit_choice' in locals() and unit_choice != 'All':
    base_master = base_master[base_master[name_col] == unit_choice]
base_master = base_master.copy()

points_all = filtered_base.dropna(subset=['Latitude','Longitude']).copy()
_selected_good_levels = [int(x) for x in (good_levels or [])]
points_good = points_all[points_all['Level'].isin(_selected_good_levels)].copy() if _selected_good_levels else points_all.iloc[0:0].copy()

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
# Export stats
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

# 🌐 Online enrichment (keep position)
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
        if not name: continue
        lat_e = e.get("lat") or (e.get("center") or {}).get("lat")
        lon_e = e.get("lon") or (e.get("center") or {}).get("lon")
        if lat_e is None or lon_e is None: continue
        addr = ", ".join(filter(None, [
            tags.get("addr:housenumber"), tags.get("addr:street"),
            tags.get("addr:city"), tags.get("addr:state"), tags.get("addr:postcode")
        ]))
        text = f"{name} {addr}"
        if BLACK_RE.search(text): continue
        if st.session_state.get("hvac_only", False) and (not HVAC_RE.search(text) and tags.get("craft") != "hvac"):
            continue
        out.append({
            "Name": name, "Address": addr if addr else pd.NA,
            "Latitude": float(lat_e), "Longitude": float(lon_e),
            "Level": np.nan, "ZIP": tags.get("addr:postcode"),
            "State": pd.NA, "City": pd.NA, "County": pd.NA,
            "Source": "web-osm", "SourceTag": tags.get("craft",""), "SourceId": str(e.get("id")),
            "Rating": pd.NA, "UserRatingsTotal": pd.NA,
        })
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def _gplaces_nearby_once(params, log=None):
    base = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    try:
        r = requests.get(base, params=params, timeout=20)
        j = r.json()
    except Exception as e:
        if log: log(f"Google request error: {e}")
        return None
    status = j.get("status", "")
    if status != "OK":
        msg = j.get("error_message", "")
        if log: log(f"Google returned: {status}{(' - ' + msg) if msg else ''}")
        return {"status": status, "results": j.get("results", []), "next_page_token": j.get("next_page_token")}
    return {"status": "OK", "results": j.get("results", []), "next_page_token": j.get("next_page_token")}

@st.cache_data(show_spinner=False, ttl=900)
def _cache_ok_signature(params_signature, page):
    return True

def fetch_google_places(lat, lng, radius_m=30000, api_key=None, hvac_only=False, log=None):
    key = api_key if api_key is not None else GOOGLE_PLACES_KEY
    if not key:
        if log: log("No Google API key detected, skip Google.")
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
        time.sleep(2.2)
    if hvac_only and not out:
        if log: log("type=hvac_contractor had no hits; try keyword fallback…")
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
    if log: log(f"Google effective results: {len(out)}")
    return out

def fetch_online_candidates_for_county(row, radius_m=30000, api_key=None, hvac_only=False, source_mode="Auto (Google first)", log=None):
    lat = float(row["cLat"]); lng = float(row["cLng"])
    items = []
    if source_mode == "Google only (faster)":
        items = fetch_google_places(lat, lng, radius_m=radius_m, api_key=api_key, hvac_only=hvac_only, log=log)
    elif source_mode == "OSM only (fallback)":
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

with st.expander("🌐 Online enrichment", expanded=False):
    st.caption("(Fetch using the currently selected State/County/City)")
    col_a, col_b, col_c = st.columns([1.2, 1, 1.2])
    with col_a:
        max_cnties = st.number_input("Max areas to fetch", 1, 200, 10, 1)
    with col_b:
        search_rad = st.slider("Radius (km)", 5, 80, 30, 5)
    with col_c:
        merge_back = st.checkbox("Merge into dataset (and map)", value=True)

    do_fetch = st.button("🔵 One-click add web technicians (blue pins)", use_container_width=True)
    online_df = pd.DataFrame()

    if do_fetch:
        log_box = st.empty()

        def log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            log_box.markdown(f"📝 **[{ts}]** {msg}")

        take_n = int(max_cnties)
        radius_m_fetch = int(search_rad * 1000)

        # 注意这里用 County/All（英文），不要再用 “郡/全部”
        base = counties_master if geo_level.startswith("County") else cities_master

        if unit_choice != 'All' and state_choice != 'All':
            rows = (
                base[(base["State"] == state_choice) & (base[name_col] == unit_choice)]
                .head(1)
                .to_dict("records")
            )
        elif state_choice != 'All':
            rows = (
                base[base["State"] == state_choice]
                .sort_values(["State", name_col])
                .head(take_n)
                .to_dict("records")
            )
        else:
            rows = (
                base.sort_values(["State", name_col])
                .head(take_n)
                .to_dict("records")
            )

        if not rows:
            st.warning("No target area selected. Choose State/County/City on the left and try again.")
        else:
            log(
                f"Start fetching {len(rows)} area(s) (R={search_rad}km, HVAC only: {st.session_state.get('hvac_only', False)}, source: {source_mode})…"
            )
            all_items = []
            for i, r0 in enumerate(rows, 1):
                unit_name = r0.get("County") if geo_level.startswith("County") else r0.get("City")
                log(f"{i}/{len(rows)}: {r0['State']} / {unit_name} — requesting…")
                try:
                    items = fetch_online_candidates_for_county(
                        r0,
                        radius_m=radius_m_fetch,
                        api_key=GOOGLE_PLACES_KEY,
                        hvac_only=st.session_state.get('hvac_only', False),
                        source_mode=source_mode,
                        log=log,
                    )
                    all_items.extend(items)
                    log(f"✓ Added {len(items)} item(s), total {len(all_items)}.")
                except Exception as e:
                    log(f"× Fetch failed: {e}")
                time.sleep(0.2)

            if all_items:
                online_df = (
                    pd.DataFrame(all_items)
                    .drop_duplicates(subset=["Source", "SourceId"], keep="first")
                    .reset_index(drop=True)
                )
                online_df["Level"] = 7
            else:
                online_df = pd.DataFrame(
                    columns=[
                        "Name",
                        "Address",
                        "Latitude",
                        "Longitude",
                        "Level",
                        "State",
                        "City",
                        "County",
                        "ZIP",
                        "Source",
                        "SourceId",
                        "Rating",
                        "UserRatingsTotal",
                    ]
                )

            if not online_df.empty:
                st.session_state["_web_new_layer"] = online_df.to_dict("records")

            buf_new = io.BytesIO()
            with pd.ExcelWriter(buf_new, engine="openpyxl") as w:
                online_df.to_excel(w, index=False, sheet_name="WebNew")
            buf_new.seek(0)

            st.download_button(
                "Download fetched list (Excel)",
                data=buf_new,
                file_name=f"web_new_workers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_web_new",
            )

            if online_df.empty:
                st.warning("Done, but 0 valid results (API returned nothing, filtered out, or Google Key not set).")
            else:
                st.success(f"Done. {len(online_df)} valid result(s).")

            if merge_back and not online_df.empty:
                cols = ["Name", "Address", "Latitude", "Longitude", "Level", "State", "City", "County", "ZIP"]
                for c in cols:
                    if c not in online_df.columns:
                        online_df[c] = pd.NA
                st.session_state.df = pd.concat([st.session_state.df, online_df[cols]], ignore_index=True)
                st.toast("Merged web-new points (Level=7) into dataset.")
                _safe_rerun()


# ======================
# Search row (under the map)
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
# Filter + matched set
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
    """
    Normalize for duplicate-address detection:
    - lower
    - full-width comma -> half-width
    - commas -> space
    - 'FL 1' -> 'fl1'
    - non-alnum -> space
    - compress spaces
    """
    if addr is None:
        return ""
    s = str(addr).strip().lower()
    s = s.replace('，', ',')            # full-width comma -> half-width
    s = s.replace(',', ' ')             # commas -> spaces
    s = re.sub(r'\bfl\s*([0-9]+)\b', r'fl\1', s)  # 'fl 1' -> 'fl1'
    s = re.sub(r'[^a-z0-9]', ' ', s)    # keep only [a-z0-9]
    s = re.sub(r'\s+', ' ', s).strip()  # compress spaces
    return s

mask = pd.Series(True, index=df.index)
if level_choice != 'All':
    mask &= (df['Level'] == level_choice)
if state_choice != 'All':
    mask &= (df['State'] == state_choice)
if geo_level.startswith("County") and unit_choice != 'All':
    mask &= (df['County'] == unit_choice)
if (not geo_level.startswith("County")) and unit_choice != 'All':
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

points = filtered.dropna(subset=['Latitude','Longitude']).copy()
if st.session_state.get("show_only_new", False):
    points = points[points['Level'].eq(7)]

# Debug panel at the bottom of sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander("🛠 Display/Debug (for troubleshooting)", expanded=False):
        st.write(f"Visible points: **{len(points)}** / Original with coords: **{df.dropna(subset=['Latitude','Longitude']).shape[0]}**")
        st.caption(f"Missing lat: {int(df['Latitude'].isna().sum())} · Missing lon: {int(df['Longitude'].isna().sum())}")
        force_show_all = st.checkbox("Ignore filters (show all points with coordinates)", value=False, key="force_show_all_cb")

if st.session_state.get("force_show_all_cb"):
    points = df.dropna(subset=['Latitude','Longitude']).copy()

if only_show_good_points:
    _sel_lvls = [int(x) for x in (good_levels or [])]
    points = points[points['Level'].isin(_sel_lvls)] if _sel_lvls else points.iloc[0:0]

# Apply “only duplicate addresses” filter (after good-level filter)
if st.session_state.get("only_dup_addr_v3", False):
    addr_series = points.apply(_full_address_from_row, axis=1).map(_norm_addr_for_dup)
    vc = addr_series.value_counts()
    dup_mask = addr_series.map(vc).fillna(0) >= 2
    points = points[dup_mask].copy()

# Build duplicate export (always recompute counts for export)
_addr_series_all = points.apply(_full_address_from_row, axis=1).map(_norm_addr_for_dup)
_vc_all = _addr_series_all.value_counts()
_dup_mask_all = _addr_series_all.map(_vc_all).fillna(0).astype(int) >= 2

dup_points_export = points[_dup_mask_all].copy()
if not dup_points_export.empty:
    dup_points_export.insert(
        0, "Duplicate Count", _addr_series_all[_dup_mask_all].map(_vc_all).astype(int).values
    )
    _cols_pref = ["Duplicate Count","Name","Address","City","State","ZIP","Level","Latitude","Longitude"]
    _cols_exist = [c for c in _cols_pref if c in dup_points_export.columns]
    _dup_buf = _build_xlsx(dup_points_export[_cols_exist], sheet_name="DuplicateAddresses")
else:
    _dup_buf = None

# Render the "export duplicates" button under the checkbox
if 'dup_export_slot' in locals():
    dup_export_slot.empty()
    if st.session_state.get("only_dup_addr_v3", False):
        if _dup_buf is not None:
            with dup_export_slot:
                st.download_button(
                    "Export duplicate addresses (Excel)",
                    data=_dup_buf,
                    file_name=f"duplicate_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_dup_excel",
                )
        else:
            with dup_export_slot:
                st.caption("(No duplicate addresses to export under the current filter)")

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
# Map drawing
# ======================
US_STATES_GEO_PATH = os.path.join(data_dir, "us_states.geojson")
LEVEL_COLORS = {
    1:'#2ecc71',  # green
    2:'#FFD700',  # gold
    3:'#FF4D4F',  # red
    4:'#FFC0CB',  # pink
    5:'#8A2BE2',  # purple
    6:'#000000',  # black
    7:'#1E90FF',  # blue (Level=7 in dataset)
}

prefer_canvas = st.session_state.get("perf_prefer_canvas", True)

# Do not rely on provider name; mount multiple tile sources to avoid CDN blocking
m = folium.Map(
    location=[37.8, -96.0],
    zoom_start=4,
    keyboard=False,
    prefer_canvas=prefer_canvas,
    tiles=None
)

# Backup 1: OSM HOT (FR)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    attr="&copy; OpenStreetMap France, HOT",
    name="OSM HOT (backup)", control=True, max_zoom=19, overlay=False
).add_to(m)

# Backup 2: OSM DE (DE)
folium.TileLayer(
    tiles="https://tile.openstreetmap.de/{z}/{x}/{y}.png",
    attr="&copy; OpenStreetMap DE",
    name="OSM DE (backup)", control=True, max_zoom=19, overlay=False
).add_to(m)

# Backup 3: Carto (use when network is good)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    attr="&copy; CARTO",
    name="Carto Positron (backup)", control=True, max_zoom=20, overlay=False
).add_to(m)

# Default OSM official
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="&copy; OpenStreetMap contributors",
    name="OSM (official)", control=True, max_zoom=19, overlay=False
).add_to(m)

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
        is_selected = (state_choice != 'All' and code == state_choice)
        return {'fillColor':'#ffffff','fillOpacity':0.0,'color':'#2563eb','weight':3.0 if is_selected else 1.8,'dashArray':None}
    gj = folium.GeoJson(
        data=states_fc, name="US States",
        style_function=style_fn, highlight_function=None,
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'])
    ).add_to(m)

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

# County/City circles
radius_m = radius_miles * 1609.34
unit_fg = folium.FeatureGroup(name=layer_name, show=True).add_to(m)

for _, r in centroids_to_plot.iterrows():
    ring_color = '#1e88e5' if bool(r['meets']) else '#9e9e9e'
    tip = (f"{'County' if geo_level.startswith('County') else 'City'}: {r.get(name_col)} "
       f"({r.get('State')}) | Good: {int(r['good_in_radius'])} / All: {int(r['all_in_radius'])}")

    folium.Circle(location=[r['cLat'], r['cLng']], radius=radius_m,
                  color=ring_color, weight=1.6, fill=True, fill_opacity=0.05,
                  tooltip=tip).add_to(unit_fg)

# Popups
POPUP_MAX_W = 520
def make_worker_popup(
    name, level, address=None, zip_code=None, distance_text="",
    contact=None, email=None, phone=None, note=None, is_cold=False, is_hot=False, **kwargs
):
    if not address:
        address = kwargs.get("state", "")
    def _has(v):
        try:
            if v is None or pd.isna(v): return False
        except Exception:
            if v is None: return False
        return str(v).strip() != ""
    icons_html = ""
    if bool(is_cold): icons_html += "<span style='font-size:18px;line-height:1'>🔧</span>"
    if bool(is_hot):  icons_html += "<span style='font-size:18px;line-height:1'>🔥</span>"

    contact_html = f"<div><b>Contact:</b> {_s(contact)}</div>" if _has(contact) else ""
    phone_html   = f"<div><b>Phone:</b> {_s(phone)}</div>"     if _has(phone)   else ""
    email_html   = f'<div><b>Email:</b> <a href="mailto:{_s(email)}" target="_blank">{_s(email)}</a></div>' if _has(email) else ""
    note_html    = f"<div><b>Note:</b> {_s(note)}</div>" if _has(note) else ""

    html = f"""
    <div style="min-width:420px; font-size:13px; line-height:1.4; white-space:normal;">
      <div><b>Name:</b> {_s(name)} {icons_html}</div>
      <div><b>Level:</b> {_s(level)}</div>
      <div><b>Address:</b> {_s(address)}</div>
      {contact_html}{phone_html}{email_html}{note_html}
      <div><b>Distance:</b> {_s(distance_text)}</div>
    </div>
    """
    return folium.Popup(html, max_width=POPUP_MAX_W)

def make_lite_popup_row(row):
    addr = _full_address_from_row(row)
    dist = popup_distance_text(row['LatAdj'], row['LngAdj'], prefer_drive=False)

    icons_html = ""
    if bool(row.get('IsColdFlag', False)): icons_html += " 🔧"
    if bool(row.get('IsHotFlag',  False)): icons_html += " 🔥"

    html = f"""
    <div style="min-width:260px; font-size:12.5px; line-height:1.35; white-space:normal;">
      <div><b>Name:</b> {_s(row.get('Name',''))}{icons_html}</div>
      <div><b>Level:</b> {_s(row.get('Level',''))}</div>
      <div><b>Address:</b> {_s(addr)}</div>
      <div><b>Distance:</b> {_s(dist)}</div>
    </div>
    """
    return folium.Popup(html, max_width=360)

# Distance/time helpers
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
    if not cust_pin:
        return "-"
    dline = haversine_miles(cust_pin["lat"], cust_pin["lng"], float(lat), float(lng))
    if prefer_drive:
        drive = osrm_drive_info(cust_pin["lat"], cust_pin["lng"], float(lat), float(lng))
        if drive:
            return f"{drive[0]:.1f} mi · {int(round(drive[1]))} min"
    return f"{dline:.1f} mi (straight-line)"

# INHOUSE-TECH check
def _is_inhouse(name: str) -> bool:
    return "INHOUSE-TECH" in str(name).upper()

# Icon factory
def _make_marker_icon(color_hex: str, larger: bool = False):
    # larger=True -> INHOUSE-TECH keeps large pin (54h), else small (28h)
    return ring_pin_icon(color_hex, size_h_px=(54 if larger else 28))

# Spread points with identical coordinates but different names
points['LatAdj'] = points['Latitude'].values
points['LngAdj'] = points['Longitude'].values

if not points.empty:
    grp = points.groupby(['Latitude','Longitude'])
    for (lat0, lng0), idxs in grp.groups.items():
        sub = points.loc[idxs]
        if sub['Name'].astype(str).nunique() > 1 and len(sub) > 1:
            k = len(sub)
            delta = 0.00035
            lat_rad = np.radians(lat0 if pd.notna(lat0) else 0.0)
            for j, idx in enumerate(sub.index):
                ang  = 2*np.pi * (j / k)
                dlat = delta * np.cos(ang)
                dlng = (delta / max(0.15, np.cos(lat_rad))) * np.sin(ang)
                points.at[idx, 'LatAdj'] = float(lat0) + dlat
                points.at[idx, 'LngAdj'] = float(lng0) + dlng

# Cluster / fast-dot modes
use_cluster = st.session_state.get("perf_use_cluster", True)
fast_mode = (not use_cluster) and st.session_state.get("perf_fast_dots", True)

# Point layer (cluster vs non-cluster)
workers_fg = folium.FeatureGroup(name="Technician points", show=True).add_to(m)
use_cluster = st.session_state.get("perf_use_cluster", True)


# === Cluster mode (robust): 每个等级一个独立集群，直接挂到 m，而不是挂到 workers_fg ===
if use_cluster:
    clusters = {}

    # 仅为当前 points 中“实际存在”的等级建层
    present_levels = (
        points['Level']
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    present_levels = sorted(present_levels)

    # 为存在的每个等级创建独立集群（直接 add_to(m)）
    for lvl in present_levels:
        col = LEVEL_COLORS.get(int(lvl), '#3388ff')
        clusters[int(lvl)] = MarkerCluster(
            name=f"Level {int(lvl)}",
            icon_create_function=f"""
            function(cluster) {{
              var count = cluster.getChildCount();
              return new L.DivIcon({{
                html: '<div style="background:{col};opacity:0.85;border-radius:20px;width:36px;height:36px;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;border:2px solid white;">'+count+'</div>',
                className: 'marker-cluster', iconSize: new L.Point(36, 36)
              }});
            }}
            """
        ).add_to(m);  # ← 关键：直接挂在地图 m，而不是 workers_fg

    # 无等级/解析失败的点，放入一个独立分组，直接挂到 m
    others_fg = folium.FeatureGroup(name="Others (no level)", show=True).add_to(m)

    # 投放点
    for _, row in points.iterrows():
        # 安全解析 Level
        lvl = None
        if not pd.isna(row['Level']):
            try:
                lvl = int(row['Level'])
            except Exception:
                lvl = None

        base_color = LEVEL_COLORS.get(lvl, '#3388ff')
        larger = _is_inhouse(row.get('Name',''))
        icon = _make_marker_icon('#1E90FF' if larger else base_color, larger=larger)

        distance_text = popup_distance_text(row['LatAdj'], row['LngAdj'], prefer_drive=False)
        popup_obj = make_worker_popup(
            name=row.get('Name',''),
            level=row.get('Level',''),
            address=_full_address_from_row(row),
            distance_text=distance_text,
            contact=row.get('Contact'),
            email=row.get('Email'),
            phone=row.get('Phone'),
            note=row.get('Note'),
            is_cold=row.get('IsColdFlag', False),
            is_hot=row.get('IsHotFlag',  False),
        )

        target_layer = clusters.get(lvl, None)
        if target_layer is None:
            target_layer = others_fg

        folium.Marker(
            location=[row['LatAdj'], row['LngAdj']],
            icon=icon,
            popup=popup_obj,
            tooltip=_s(row.get('Name',''))
        ).add_to(target_layer)



# Non-cluster mode (fast dots & static optimization)
else:
    dot_r = int(st.session_state.get("perf_fast_radius", 8))

    # 一定要先算出 n_points，再做兜底判断
    n_points = len(points)
    EXTREME_STATIC = (USE_STATIC_MAP and n_points >= int(st.session_state.get("perf_fast_threshold", 2500)))

    # 读取当前设置
    fast_mode = st.session_state.get("perf_fast_dots", True)

    # 兜底：若用户关闭快点模式且点数很大，临时启用，防止浏览器挂掉导致空白
    if (not fast_mode) and n_points > 1200:
        st.warning("Too many points without clustering; auto-enabled fast dots mode to prevent blank canvas.")
        fast_mode = True


    def _popup_for_row(row):
        return make_worker_popup(
            name=row.get('Name',''),
            level=row.get('Level',''),
            address=_full_address_from_row(row),
            distance_text=popup_distance_text(row['LatAdj'], row['LngAdj'], prefer_drive=False),
            contact=row.get('Contact'),
            email=row.get('Email'),
            phone=row.get('Phone'),
            note=row.get('Note'),
            is_cold=row.get('IsColdFlag', False),
            is_hot=row.get('IsHotFlag',  False),
        )

    if EXTREME_STATIC:
        for _, row in points.iterrows():
            lvl = int(row['Level']) if not pd.isna(row['Level']) else None
            base_color = LEVEL_COLORS.get(lvl, '#3388ff')

            if _is_inhouse(row.get('Name','')):
                icon = _make_marker_icon('#1E90FF', larger=True)
                folium.Marker(
                    location=[row['LatAdj'], row['LngAdj']],
                    icon=icon,
                    tooltip=_s(row.get('Name','')),
                    popup=_popup_for_row(row)
                ).add_to(workers_fg)
            else:
                folium.CircleMarker(
                    location=[row['LatAdj'], row['LngAdj']],
                    radius=dot_r,
                    color=base_color,
                    fill=True,
                    fill_color=base_color, 
                    fill_opacity=0.85,
                    weight=0,
                    tooltip=_s(row.get('Name','')),
                    popup=_popup_for_row(row)
                ).add_to(workers_fg)
        st.caption(f"🧩 Extreme static optimization: {n_points:,} points (dots still have full popups; HTML may be large)")

    elif fast_mode:
        for _, row in points.iterrows():
            lvl = int(row['Level']) if not pd.isna(row['Level']) else None
            base_color = LEVEL_COLORS.get(lvl, '#3388ff')

            if _is_inhouse(row.get('Name','')):
                icon = _make_marker_icon('#1E90FF', larger=True)
                folium.Marker(
                    location=[row['LatAdj'], row['LngAdj']],
                    icon=icon,
                    tooltip=_s(row.get('Name','')),
                    popup=_popup_for_row(row)
                ).add_to(workers_fg)
            else:
                folium.CircleMarker(
                    location=[row['LatAdj'], row['LngAdj']],
                    radius=dot_r,
                    color=base_color,
                    fill=True,
                    fill_color=base_color, 
                    fill_opacity=0.85,
                    weight=0,
                    tooltip=_s(row.get('Name','')),
                    popup=_popup_for_row(row)
                ).add_to(workers_fg)

    else:
        for _, row in points.iterrows():
            lvl = int(row['Level']) if not pd.isna(row['Level']) else None
            base_color = LEVEL_COLORS.get(lvl, '#3388ff')
            larger = _is_inhouse(row.get('Name',''))
            icon = _make_marker_icon('#1E90FF' if larger else base_color, larger=larger)
            folium.Marker(
                location=[row['LatAdj'], row['LngAdj']],
                icon=icon,
                popup=_popup_for_row(row),
                tooltip=_s(row.get('Name',''))
            ).add_to(workers_fg)

# ======================
# Search matches: red flags
# ======================
def render_hit_flags(map_obj, matched_df):
    if matched_df is None or matched_df.empty:
        return
    for _, r in matched_df.iterrows():
        dist_txt = popup_distance_text(r['Latitude'], r['Longitude'], prefer_drive=True)
        popup_obj = make_worker_popup(
            name=r.get('Name',''),
            level=r.get('Level',''),
            address=_full_address_from_row(r),
            zip_code=None,
            distance_text=dist_txt,
            contact=r.get('Contact'),
            email=r.get('Email'),
            phone=r.get('Phone'),
            note=r.get('Note'),
            is_cold=r.get('IsColdFlag', False),
            is_hot=r.get('IsHotFlag',  False),
        )
        folium.Marker(
            location=[float(r['Latitude']), float(r['Longitude'])],
            icon=big_flag_icon(size_px=42),
            tooltip=f"🔎 Match: {_s(r.get('Name',''))}",
            popup=popup_obj,
            z_index_offset=10000
        ).add_to(map_obj)

if search_active:
    render_hit_flags(m, matched)

# Customer pin
if cust_pin:
    p = cust_pin
    folium.Marker(
        location=[float(p["lat"]), float(p["lng"])],
        icon=customer_pin_icon(size_px=42),
        tooltip=f"📍 {p.get('formatted','Customer address')}",
        popup=f"<b>Customer:</b> {_s(p.get('formatted'))}",
        z_index_offset=12000
    ).add_to(m)

# Web-new layer (brown small pins)
if "_web_new_layer" in st.session_state:
    add_fg = folium.FeatureGroup(name="Web-new techs (fetched)", show=True).add_to(m)
    for r in st.session_state.pop("_web_new_layer"):
        name = _s(r.get("Name"))
        dist_txt = popup_distance_text(r.get("Latitude"), r.get("Longitude"), prefer_drive=False)
        popup_obj = make_worker_popup(name, r.get("Level","7"), _s(r.get("Address","")), None, dist_txt)
        folium.Marker(
            location=[float(r["Latitude"]), float(r["Longitude"])],
            icon=blue_wrench_icon(size_px=18),  # smaller pin
            tooltip=f"New: {name}",
            popup=popup_obj
        ).add_to(add_fg)

# Initial / smart fit
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
    if (state_choice != 'All') and state_bnds:
        map_obj.fit_bounds(state_bnds); return
    map_obj.fit_bounds(nat_bnds)

fit_initial_or_search(m, CONUS_BOUNDS, selected_bounds, matched if 'matched' in locals() else None, search_active)

# Legend
# Legend（竖列；去掉 "Level" 前缀）
lvl_order = sorted(LEVEL_COLORS.keys())
lvl_counts = {lvl: int(points['Level'].eq(lvl).sum()) for lvl in lvl_order}

rows_html = "".join([
    (
        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
        f"<span style='display:inline-block;width:10px;height:10px;background:{LEVEL_COLORS[lvl]};"
        f"border-radius:2px{(';border:1px solid #eee' if lvl==6 else '')}'></span>"
        f"<span>{lvl}</span>"  # ← 这里原来是 'Level {lvl}'，已改为只显示数字
        f"<span style='margin-left:auto'>{lvl_counts.get(lvl,0)}</span>"
        f"</div>"
    )
    for lvl in lvl_order
])

legend_html = f"""
<div style="
  position: fixed; top: 10px; right: 12px; z-index: 9999;
  background: #fff; padding: 6px 8px; border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,.20); font-size: 12px; line-height: 1.4;
  width: max-content; max-width: 220px; border: 1px solid rgba(0,0,0,.08);
">
  <div style="font-weight:600; margin-bottom:6px;">Level colors</div>
  {rows_html}
  <div style="margin-top:6px; border-top:1px dashed #e5e7eb; padding-top:6px; font-weight:600;">
    Total: {int(points.shape[0])}
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))



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
            map_height = st.session_state.get("map_height", 760)
            st_folium(m, use_container_width=True, height=map_height)
            st.caption("✅ Rendered: interactive (st_folium)")
    except Exception as e:
        st.error(f"Map render failed: {e}")
        st.stop()

render_map_once(m)