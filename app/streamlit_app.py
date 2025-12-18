import json
from datetime import datetime, date, time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# Mapa
import folium
from streamlit_folium import st_folium


# =========================
# Config
# =========================
DEFAULT_API_URL = "http://localhost:8000"
NYC_CENTER = (40.7580, -73.9855)  # Times Square aprox


# =========================
# Helpers: API
# =========================
@st.cache_resource
def get_session() -> requests.Session:
    s = requests.Session()
    return s


@st.cache_data(ttl=15)
def api_health(api_url: str) -> Dict[str, Any]:
    s = get_session()
    r = s.get(f"{api_url}/health", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def api_models(api_url: str) -> List[Dict[str, Any]]:
    s = get_session()
    r = s.get(f"{api_url}/models", timeout=8)
    r.raise_for_status()
    data = r.json()
    # Asegura lista
    if isinstance(data, dict) and "models" in data:
        return data["models"]
    if isinstance(data, list):
        return data
    return []


def api_predict(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    s = get_session()
    r = s.post(f"{api_url}/predict", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


# =========================
# Helpers: Geo
# =========================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    # No numpy para evitar dependencias adicionales
    import math
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def km_to_miles(km: float) -> float:
    return km * 0.621371


# =========================
# UI: Map selector
# =========================
def build_map(
    center: Tuple[float, float],
    pickup: Optional[Tuple[float, float]],
    dropoff: Optional[Tuple[float, float]],
    zoom: int = 12,
) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    # Marcadores
    if pickup is not None:
        folium.Marker(
            location=pickup,
            tooltip="Pickup",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

    if dropoff is not None:
        folium.Marker(
            location=dropoff,
            tooltip="Dropoff",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    # Línea recta entre puntos
    if pickup is not None and dropoff is not None:
        folium.PolyLine(
            locations=[pickup, dropoff],
            weight=5,
        ).add_to(m)

        # Ajustar bounds para que se vea todo
        m.fit_bounds([pickup, dropoff])

    return m


def parse_last_click(map_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # streamlit_folium retorna un dict con "last_clicked": {"lat":..., "lng":...}
    if not map_data:
        return None
    lc = map_data.get("last_clicked")
    if not lc:
        return None
    lat = lc.get("lat")
    lng = lc.get("lng")
    if lat is None or lng is None:
        return None
    return (float(lat), float(lng))


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="NYC Taxi Fare Prediction", page_icon="🚕", layout="centered")

st.title("NYC Taxi Fare Prediction 🚕💵")
st.caption("Select pickup and dropoff on the map, then estimate the final total cost via the API.")

# Sidebar: API settings
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_URL)

# Health check
health_ok = False
try:
    health = api_health(api_url)
    health_ok = True
    st.sidebar.success(f"API online ({health})")
except Exception as e:
    st.sidebar.error(f"API offline ({e})")

# Models list
models = []
selected_model = None
if health_ok:
    try:
        models = api_models(api_url)
    except Exception:
        models = []

    # Selector flexible: si tu /models no existe o devuelve vacío, no bloquea
    if models:
        # Intentamos inferir campos comunes
        def label(m: Dict[str, Any]) -> str:
            for k in ("model_version", "model_name", "name", "id"):
                if k in m and m[k] is not None:
                    return str(m[k])
            return "model"

        options = [label(m) for m in models]
        selected_model = st.sidebar.selectbox("Model", options)
    else:
        st.sidebar.info("No /models available (optional). Using default model on /predict.")

# Session state for points
if "pickup" not in st.session_state:
    st.session_state.pickup = None
if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    if st.button("Set pickup (next click)"):
        st.session_state._next_click_role = "pickup"
with colB:
    if st.button("Set dropoff (next click)"):
        st.session_state._next_click_role = "dropoff"
with colC:
    if st.button("Reset points"):
        st.session_state.pickup = None
        st.session_state.dropoff = None
        st.session_state._next_click_role = "pickup"

# Default role
if "_next_click_role" not in st.session_state:
    st.session_state._next_click_role = "pickup"

st.subheader("1) Pick locations")
m = build_map(NYC_CENTER, st.session_state.pickup, st.session_state.dropoff)
map_data = st_folium(m, height=420, width=None)

last_click = parse_last_click(map_data)
if last_click is not None:
    role = st.session_state._next_click_role
    if role == "pickup":
        st.session_state.pickup = last_click
        st.session_state._next_click_role = "dropoff"
    else:
        st.session_state.dropoff = last_click
        st.session_state._next_click_role = "pickup"

pickup = st.session_state.pickup
dropoff = st.session_state.dropoff

# Derived distance
distance_miles = None
if pickup and dropoff:
    dist_km = haversine_km(pickup[0], pickup[1], dropoff[0], dropoff[1])
    distance_miles = km_to_miles(dist_km)

st.divider()

st.subheader("2) Trip details")
col1, col2 = st.columns(2)

with col1:
    d = st.date_input("Pickup date", value=date(2022, 5, 15))
with col2:
    t = st.time_input("Pickup time", value=time(14, 30))

pickup_datetime = datetime.combine(d, t)

col3, col4, col5 = st.columns(3)
with col3:
    passenger_count = st.number_input("Passenger count", min_value=1, max_value=8, value=1, step=1)
with col4:
    vendor_id = st.selectbox("Vendor ID", options=[1, 2], index=0)
with col5:
    payment_type = st.selectbox("Payment type", options=[1, 2, 3, 4, 5, 6], index=0)

# Show computed distance
if distance_miles is not None:
    st.info(f"Computed straight-line distance: **{distance_miles:.2f} miles** (Haversine)")
else:
    st.warning("Click **pickup** and **dropoff** on the map to compute distance.")

# Payload preview
payload: Dict[str, Any] = {
    "pickup_datetime": pickup_datetime.isoformat(),
    "passenger_count": int(passenger_count),
    "vendor_id": int(vendor_id),
    "payment_type": int(payment_type),
}

if pickup:
    payload["pickup_latitude"] = float(pickup[0])
    payload["pickup_longitude"] = float(pickup[1])

if dropoff:
    payload["dropoff_latitude"] = float(dropoff[0])
    payload["dropoff_longitude"] = float(dropoff[1])

# OPTIONAL: si tu API soporta seleccionar modelo (depende de tu backend)
# Si no lo soporta, no pasa nada, pero podrías quitarlo.
if selected_model is not None:
    payload["model_version"] = selected_model

with st.expander("Debug: request payload"):
    st.code(json.dumps(payload, indent=2), language="json")

st.divider()

st.subheader("3) Predict")
can_predict = health_ok and pickup is not None and dropoff is not None

if not health_ok:
    st.error("API is offline. Start FastAPI first.")
elif not can_predict:
    st.error("Missing pickup/dropoff points. Select both on the map.")

if st.button("Predict fare", disabled=not can_predict):
    try:
        res = api_predict(api_url, payload)

        # Campos esperados según tu doc:
        predicted_fare = res.get("predicted_fare")
        predicted_duration = res.get("predicted_duration")

        if predicted_fare is None:
            st.error(f"API response missing 'predicted_fare': {res}")
        else:
            st.success(f"Estimated total cost: **${float(predicted_fare):.2f}**")

        if predicted_duration is not None:
            st.info(f"Estimated duration: **{float(predicted_duration):.1f} min**")

        meta = []
        if "model_version" in res:
            meta.append(f"Model: {res['model_version']}")
        if "timestamp" in res:
            meta.append(f"Timestamp: {res['timestamp']}")
        if meta:
            st.caption(" | ".join(meta))

        with st.expander("Raw API response"):
            st.json(res)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
