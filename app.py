#!/usr/bin/env python3
# pip install flask flask-cors cdsapi xarray netCDF4 numpy requests

import os, math, pathlib, re, shutil
from datetime import datetime, timedelta, date

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import cdsapi
import xarray as xr
import numpy as np

# ------------------ ADS / CAMS ------------------
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
PRIMARY_DATASET = "cams-europe-air-quality-forecasts"
POLLEN_DATASET  = "cams-europe-pollen-forecasts"  # тут чаще лежат «редкие» виды

# ключ: переменная окружения ADS_API_KEY имеет приоритет
HARDCODED_ADS_TOKEN = "911411d0-2402-48bf-b07b-fdb19c785d6b"

def ensure_client():
    token = os.environ.get("ADS_API_KEY", HARDCODED_ADS_TOKEN).strip()
    if not token:
        raise RuntimeError("ADS token missing")
    return cdsapi.Client(url=ADS_URL, key=token, quiet=True, verify=True)

# ------------------ переменные ------------------
BASE_VARS = ["birch_pollen", "grass_pollen", "olive_pollen", "ragweed_pollen"]
EXTENDED_VARS = BASE_VARS + [
    "alder_pollen","hazel_pollen","ash_pollen","oak_pollen",
    "mugwort_pollen","plane_pollen",
    "willow_pollen","poplar_pollen","pine_pollen",
    "cypress_pollen","juniper_pollen",
    "plantain_pollen","sorrel_pollen","nettle_pollen","chenopodium_pollen",
]

ALIASES = {
    "Birch":["birch_pollen","bpg","bpg_conc","bpa","bpa_conc","betula_pollen"],
    "Grass":["grass_pollen","gpg","gpg_conc","gpa","gpa_conc","poaceae_pollen"],
    "Olive":["olive_pollen","opg","opg_conc","opa","opa_conc","olea_pollen"],
    "Ragweed":["ragweed_pollen","rwpg","rwpg_conc","rwpa","rwpa_conc","ambrosia_pollen"],
    "Alder":["alder_pollen","alnus_pollen"],
    "Hazel":["hazel_pollen","corylus_pollen"],
    "Ash":["ash_pollen","fraxinus_pollen"],
    "Oak":["oak_pollen","quercus_pollen"],
    "Mugwort":["mugwort_pollen","artemisia_pollen","apg","apg_conc","apa","apa_conc"],
    "Plane":["plane_pollen","platanus_pollen"],
    "Willow":["willow_pollen","salix_pollen"],
    "Poplar":["poplar_pollen","populus_pollen"],
    "Pine":["pine_pollen","pinus_pollen"],
    "Cypress":["cypress_pollen","cupressaceae_pollen"],
    "Juniper":["juniper_pollen","juniperus_pollen"],
    "Plantain":["plantain_pollen","plantago_pollen"],
    "Sorrel":["sorrel_pollen","rumex_pollen"],
    "Nettle":["nettle_pollen","urticaceae_pollen"],
    "Chenopodium":["chenopodium_pollen"],
}

SPECIES_NORMALIZE = {
    "betula":"Birch","poaceae":"Grass","olea":"Olive","ambrosia":"Ragweed",
    "alnus":"Alder","corylus":"Hazel","fraxinus":"Ash","quercus":"Oak",
    "artemisia":"Mugwort","mugwort":"Mugwort",
    "platanus":"Plane","salix":"Willow","populus":"Poplar",
    "pinus":"Pine","cupressaceae":"Cypress","juniperus":"Juniper",
    "plantago":"Plantain","rumex":"Sorrel","urticaceae":"Nettle","chenopodium":"Chenopodium",
    "birch":"Birch","grass":"Grass","olive":"Olive","ragweed":"Ragweed",
    "alder":"Alder","hazel":"Hazel","ash":"Ash","oak":"Oak",
    "plane":"Plane","willow":"Willow","poplar":"Poplar","pine":"Pine",
    "cypress":"Cypress","juniper":"Juniper","plantain":"Plantain","sorrel":"Sorrel","nettle":"Nettle",
}

UNIT = "grains/m³"
FORECAST_MAX_H = 96  # 4 суток × 24ч

# ------------------ утилиты ------------------
def classify_level(v: float) -> str:
    if v is None or not np.isfinite(v): return "Very Low"
    if v >= 90:  return "Very High"
    if v >= 61:  return "High"
    if v >= 31:  return "Medium"
    if v >= 11:  return "Low"
    return "Very Low"

def build_area(lat: float, lon: float, dlat=0.5, dlon=0.5):
    north = min(72.0, lat + dlat)
    south = max(30.0, lat - dlat)
    west  = max(-25.0, lon - dlon)
    east  = min(45.0,  lon + dlon)
    return [round(north,2), round(west,2), round(south,2), round(east,2)]

def nearest_idx(values, target):
    arr = np.asarray(values); return int(np.argmin(np.abs(arr - target)))

def approx_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111.32
    x = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1)
    return round(km_per_deg * math.sqrt(x*x + y*y), 2)

def lead_from_time_str(tstr: str) -> int:
    hh = max(0, min(24, int(tstr.split(":")[0])))
    return min(24, int(round(hh / 3.0) * 3))

def dt64_to_str(dt64: np.datetime64) -> str:
    try:
        ts = dt64.astype("datetime64[s]").astype("int64")
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None

def open_dataset_robust(path: pathlib.Path):
    try:
        return xr.open_dataset(path, decode_times=True, decode_timedelta=True).squeeze(drop=True)
    except TypeError:
        return xr.open_dataset(path, decode_times=True).squeeze(drop=True)
    except Exception:
        return xr.open_dataset(path, decode_times=False).squeeze(drop=True)

def pick_coord_names(ds):
    lat_candidates = ["latitude","lat","Latitude","y"]
    lon_candidates = ["longitude","lon","Longitude","x"]
    lat_name = next((n for n in lat_candidates if n in ds.coords), None)
    lon_name = next((n for n in lon_candidates if n in ds.coords), None)
    if not lat_name or not lon_name:
        coords = list(ds.coords)
        if len(coords) >= 2:
            lat_name, lon_name = coords[-2], coords[-1]
    return lat_name, lon_name

def guess_species(varname: str, attrs: dict) -> str | None:
    vn = varname.lower()
    for friendly, names in ALIASES.items():
        if vn in {n.lower() for n in names}:
            return friendly
    lname = (attrs.get("long_name","") or "").lower()
    m = re.search(r"([a-z]+)\s+pollen", lname)
    if m:
        cand = m.group(1)
        if cand in SPECIES_NORMALIZE: return SPECIES_NORMALIZE[cand]
    units = (attrs.get("units","") or "").lower()
    if "grain" in units:
        base = re.sub(r"(_pollen|_conc|_pg_conc|_pg|_pa_conc|_pa)$","",vn)
        base = base.replace("_"," ").strip().split()[0]
        return SPECIES_NORMALIZE.get(base)
    if "pollen" in vn:
        base = vn.replace("pollen","").replace("_"," ").strip().split()[0]
        return SPECIES_NORMALIZE.get(base)
    return None

def collect_leads(ds):
    if "leadtime_hour" in ds.coords:
        lt = ds["leadtime_hour"].values
        if np.issubdtype(ds["leadtime_hour"].dtype, np.timedelta64):
            lt = lt.astype("timedelta64[h]").astype(int)
        return list(map(float, np.asarray(lt)))
    if "valid_time" in ds.coords and np.issubdtype(ds["valid_time"].dtype, np.datetime64):
        base = np.asarray(ds["valid_time"].values)[0]
        hrs = (np.asarray(ds["valid_time"].values) - base).astype("timedelta64[h]").astype(float)
        return list(hrs)
    if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        base = np.asarray(ds["time"].values)[0]
        hrs = (np.asarray(ds["time"].values) - base).astype("timedelta64[h]").astype(float)
        return list(hrs)
    return [0,3,6,9,12,15,18,21,24]

def interp_to_hourly(leads_hours, vals, total_hours):
    """Линейная интерполяция на сетку 0..total_hours с шагом 1ч."""
    grid = np.arange(0, total_hours+1, 1, dtype=float)
    arr_leads = np.asarray(leads_hours, dtype=float)
    arr_vals  = np.asarray([np.nan if v is None else float(v) for v in vals], dtype=float)
    mask = np.isfinite(arr_vals) & np.isfinite(arr_leads)
    if mask.sum() == 0:
        return [None for _ in grid]
    if mask.sum() == 1:
        fill = float(arr_vals[mask][0]); return [fill for _ in grid]
    x = arr_leads[mask]; y = arr_vals[mask]
    y_interp = np.interp(grid, x, y, left=y[0], right=y[-1])
    return [float(v) if np.isfinite(v) else None for v in y_interp]

# ------------------ парсинг ------------------
def extract_snapshot(path, lat, lon, date_str, lead):
    ds = open_dataset_robust(path); latn,lonn = pick_coord_names(ds)
    if not latn or not lonn: ds.close(); return {}, {}
    yi = nearest_idx(ds[latn].values, lat); xi = nearest_idx(ds[lonn].values, lon)

    valid_str = None
    if "valid_time" in ds.coords and np.issubdtype(ds["valid_time"].dtype, np.datetime64):
        valid_str = dt64_to_str(np.asarray(ds["valid_time"].values).item())
    elif "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        valid_str = dt64_to_str(np.asarray(ds["time"].values).item())
    if not valid_str:
        valid_str = (datetime.fromisoformat(date_str) + timedelta(hours=lead)).strftime("%Y-%m-%d %H:%M")

    results, used, detected = {}, {}, {}
    for v in ds.data_vars:
        da = ds[v]
        if not {latn,lonn}.issubset(set(da.dims)): continue
        friendly = guess_species(v, da.attrs)
        if not friendly: continue
        detected[v] = da.attrs.get("long_name","")
        slc = da
        for d in ("time","valid_time","leadtime_hour","level"):
            if d in slc.dims and slc.sizes[d] == 1:
                slc = slc.isel({d:0})
        val = float(slc.isel({latn: yi, lonn: xi}).values)
        if not np.isfinite(val): continue
        results[friendly] = {"value": val, "level": classify_level(val), "unit": UNIT}
        used[friendly] = v

    grid_lat = float(ds[latn].values[yi]); grid_lon = float(ds[lonn].values[xi])
    ds.close()
    meta = {
        "grid_distance_km": approx_km(lat, lon, grid_lat, grid_lon),
        "grid_point": {"lat": grid_lat, "lon": grid_lon},
        "dataset_vars_used": used,
        "detected_vars_all": detected,
        "valid_str": valid_str,
    }
    return results, meta

def extract_series_hourly_for_hours(path, lat, lon, base_date_str, total_hours):
    ds = open_dataset_robust(path); latn,lonn = pick_coord_names(ds)
    if not latn or not lonn: ds.close(); return [], {}, {}
    yi = nearest_idx(ds[latn].values, lat); xi = nearest_idx(ds[lonn].values, lon)

    leads = collect_leads(ds)
    # ограничим входные лиды максимумом файла
    leads = [float(h) for h in leads if 0 <= float(h) <= total_hours]

    base_dt = datetime.fromisoformat(base_date_str)
    times_hourly = [(base_dt + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
                    for h in range(0, total_hours+1)]

    values_hourly, used = {}, {}
    for v in ds.data_vars:
        da = ds[v]
        if not {latn,lonn}.issubset(set(da.dims)): continue
        friendly = guess_species(v, da.attrs)
        if not friendly: continue
        slc = da
        if "level" in slc.dims and slc.sizes["level"] == 1:
            slc = slc.isel(level=0)
        if any(d in slc.dims for d in ("time","valid_time","leadtime_hour")):
            slc = slc.isel({latn: yi, lonn: xi})
            vec = np.asarray(slc).astype(float)
            vec_list = [float(x) if np.isfinite(x) else None for x in vec][:len(leads)]
            values_hourly[friendly] = interp_to_hourly(leads, vec_list, total_hours)
        else:
            val = float(slc.isel({latn: yi, lonn: xi}).values)
            values_hourly[friendly] = [val for _ in range(total_hours+1)]
        used[friendly] = v

    grid_lat = float(ds[latn].values[yi]); grid_lon = float(ds[lonn].values[xi])
    ds.close()
    meta = {
        "grid_distance_km": approx_km(lat, lon, grid_lat, grid_lon),
        "grid_point": {"lat": grid_lat, "lon": grid_lon},
        "leads_input_hours": leads,
        "time_dim": "hourly_interpolated",
        "dataset_vars_used": used,
    }
    return times_hourly, values_hourly, meta

# ------------------ CDS загрузка ------------------
def retrieve(client, dataset_name, req, target, variables):
    try:
        if not target.exists():
            client.retrieve(dataset_name, {**req, "variable": variables}, str(target))
        return "extended", target
    except Exception:
        base_target = target.with_name(target.stem + "_base.nc")
        if not base_target.exists():
            client.retrieve(dataset_name, {**req, "variable": BASE_VARS}, str(base_target))
        return "base", base_target

# ------------------ merge helpers ------------------
def merge_snapshot(a_res, a_meta, b_res, b_meta):
    out = dict(a_res)
    for sp,info in b_res.items():
        if sp not in out:
            out[sp] = info
    meta = {
        "dataset_vars_used": {**(a_meta.get("dataset_vars_used") or {}), **(b_meta.get("dataset_vars_used") or {})},
        "detected_vars_all": {**(a_meta.get("detected_vars_all") or {}), **(b_meta.get("detected_vars_all") or {})},
        "grid_distance_km": a_meta.get("grid_distance_km") or b_meta.get("grid_distance_km"),
        "grid_point": a_meta.get("grid_point") or b_meta.get("grid_point"),
        "valid_str": a_meta.get("valid_str") or b_meta.get("valid_str"),
        "datasets_combined": True,
    }
    return out, meta

def merge_series(a_times, a_vals, a_meta, b_times, b_vals, b_meta):
    times = a_times or b_times
    out = dict(a_vals)
    for sp, arr in b_vals.items():
        if sp not in out:
            out[sp] = arr
    meta = {
        "dataset_vars_used": {**(a_meta.get("dataset_vars_used") or {}), **(b_meta.get("dataset_vars_used") or {})},
        "grid_distance_km": a_meta.get("grid_distance_km") or b_meta.get("grid_distance_km"),
        "grid_point": a_meta.get("grid_point") or b_meta.get("grid_point"),
        "leads_input_hours": a_meta.get("leads_input_hours") or b_meta.get("leads_input_hours"),
        "datasets_combined": True,
    }
    return times, out, meta

# ------------------ Flask ------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.get("/api/health")
def health():
    try:
        _ = ensure_client()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/geocode")
def geocode():
    q = request.args.get("location","").strip()
    if not q: return jsonify({"results":[]})
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "jsonv2", "limit": 8},
            headers={"User-Agent": "cams-pollen-demo/1.7"},
            timeout=20,
        )
        r.raise_for_status()
        out=[]
        for item in r.json():
            name=item.get("display_name",""); country=None
            if isinstance(item.get("address"), dict): country=item["address"].get("country")
            out.append({"name":name,"country":country,"lat":float(item["lat"]),"lon":float(item["lon"])})
        return jsonify({"results": out})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 502

# ---------- /api/pollen (как было) ----------
@app.get("/api/pollen")
def api_pollen():
    try:
        date_str = request.args.get("date")
        time_str = request.args.get("time","12:00")
        if not date_str: return jsonify({"status":"error","message":"Missing date=YYYY-MM-DD"}), 400

        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        loc_name = request.args.get("location","").strip() or None

        if lat is None or lon is None:
            if not loc_name: return jsonify({"status":"error","message":"Provide lat/lon or location"}), 400
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": loc_name, "format":"jsonv2", "limit":1},
                             headers={"User-Agent":"cams-pollen-demo/1.7"}, timeout=20)
            r.raise_for_status(); items=r.json()
            if not items: return jsonify({"status":"error","message":"Could not geocode location"}), 400
            lat=float(items[0]["lat"]); lon=float(items[0]["lon"])

        if not (30.0 <= lat <= 72.0 and -25.0 <= lon <= 45.0):
            return jsonify({"status":"error","message":"Requested point is outside CAMS Europe domain"}), 400

        lead = lead_from_time_str(time_str)
        area = build_area(lat, lon, 0.5, 0.5)
        req_common = {"model":["ensemble"], "date":f"{date_str}/{date_str}", "format":"netcdf",
                      "type":["forecast"], "time":["00:00"], "leadtime_hour":[str(lead)], "level":["0"], "area": area}

        cache = pathlib.Path("cache"); cache.mkdir(exist_ok=True)
        client = ensure_client()

        # primary
        t1 = cache / f"cams_{date_str}_+{lead:02d}h_{lat:.2f}_{lon:.2f}.nc"
        mode1, path1 = retrieve(client, PRIMARY_DATASET, req_common, t1, EXTENDED_VARS)
        res1, meta1 = extract_snapshot(path1, lat, lon, date_str, lead)

        # pollen dataset (дополнение видами)
        t2 = cache / f"pollen_{date_str}_+{lead:02d}h_{lat:.2f}_{lon:.2f}.nc"
        try:
            mode2, path2 = retrieve(client, POLLEN_DATASET, req_common, t2, EXTENDED_VARS)
            res2, meta2 = extract_snapshot(path2, lat, lon, date_str, lead)
        except Exception:
            res2, meta2, mode2 = {}, {}, "none"

        results, meta = merge_snapshot(res1, meta1, res2, meta2)

        loc = {"lat": float(lat), "lon": float(lon)}
        if loc_name: loc["name"] = loc_name

        payload = {
            "status":"success","location":loc,
            "datetime": meta.get("valid_str", f"{date_str} {time_str}"),
            "pollen": {
                **results,
                "_metadata": {
                    **meta, "bbox": area, "leadtime_hour": lead,
                    "request_mode_primary": mode1, "request_mode_pollen": mode2,
                    "datasets_queried": [PRIMARY_DATASET, POLLEN_DATASET],
                    "species_count": len(results),
                }
            }
        }
        return jsonify(payload)

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

# ---------- /api/pollen_series (1 день, 0..24ч) ----------
@app.get("/api/pollen_series")
def api_pollen_series():
    try:
        date_str = request.args.get("date")
        if not date_str: return jsonify({"status":"error","message":"Missing date=YYYY-MM-DD"}), 400

        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        loc_name = request.args.get("location","").strip() or None

        if lat is None or lon is None:
            if not loc_name: return jsonify({"status":"error","message":"Provide lat/lon or location"}), 400
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": loc_name, "format":"jsonv2", "limit":1},
                             headers={"User-Agent":"cams-pollen-demo/1.7"}, timeout=20)
            r.raise_for_status(); items=r.json()
            if not items: return jsonify({"status":"error","message":"Could not geocode location"}), 400
            lat=float(items[0]["lat"]); lon=float(items[0]["lon"])

        if not (30.0 <= lat <= 72.0 and -25.0 <= lon <= 45.0):
            return jsonify({"status":"error","message":"Requested point is outside CAMS Europe domain"}), 400

        leads = [str(h) for h in (0,3,6,9,12,15,18,21,24)]
        area = build_area(lat, lon, 0.5, 0.5)
        req_common = {"model":["ensemble"], "date":f"{date_str}/{date_str}", "format":"netcdf",
                      "type":["forecast"], "time":["00:00"], "leadtime_hour":leads, "level":["0"], "area": area}

        cache = pathlib.Path("cache"); cache.mkdir(exist_ok=True)
        client = ensure_client()

        # primary
        t1 = cache / f"cams_{date_str}_series_{lat:.2f}_{lon:.2f}.nc"
        mode1, path1 = retrieve(client, PRIMARY_DATASET, req_common, t1, EXTENDED_VARS)
        times1, vals1, meta1 = extract_series_hourly_for_hours(path1, lat, lon, date_str, 24)

        # pollen
        t2 = cache / f"pollen_{date_str}_series_{lat:.2f}_{lon:.2f}.nc"
        try:
            mode2, path2 = retrieve(client, POLLEN_DATASET, req_common, t2, EXTENDED_VARS)
            times2, vals2, meta2 = extract_series_hourly_for_hours(path2, lat, lon, date_str, 24)
        except Exception:
            times2, vals2, meta2, mode2 = [], {}, {}, "none"

        times, values, meta = merge_series(times1, vals1, meta1, times2, vals2, meta2)
        loc = {"lat": float(lat), "lon": float(lon)}
        if loc_name: loc["name"] = loc_name

        return jsonify({
            "status":"success","location":loc,
            "times": times, "unit": UNIT, "values": values,
            "metadata": { **(meta or {}), "request_mode_primary": mode1, "request_mode_pollen": mode2,
                          "datasets_queried": [PRIMARY_DATASET, POLLEN_DATASET] }
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

# ---------- /api/pollen_range (любой диапазон + авто-прогноз до 4д) ----------
@app.get("/api/pollen_range")
def api_pollen_range():
    try:
        start_str = request.args.get("start")
        end_str   = request.args.get("end")
        if not start_str or not end_str:
            return jsonify({"status":"error","message":"Missing start=YYYY-MM-DD and end=YYYY-MM-DD"}), 400

        d0 = datetime.fromisoformat(start_str).date()
        d1 = datetime.fromisoformat(end_str).date()
        if d1 < d0:
            d0, d1 = d1, d0  # меняем местами

        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        loc_name = request.args.get("location","").strip() or None

        if lat is None or lon is None:
            if not loc_name: return jsonify({"status":"error","message":"Provide lat/lon or location"}), 400
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": loc_name, "format":"jsonv2", "limit":1},
                             headers={"User-Agent":"cams-pollen-demo/1.7"}, timeout=20)
            r.raise_for_status(); items=r.json()
            if not items: return jsonify({"status":"error","message":"Could not geocode location"}), 400
            lat=float(items[0]["lat"]); lon=float(items[0]["lon"])

        if not (30.0 <= lat <= 72.0 and -25.0 <= lon <= 45.0):
            return jsonify({"status":"error","message":"Requested point is outside CAMS Europe domain"}), 400

        cache = pathlib.Path("cache"); cache.mkdir(exist_ok=True)
        client = ensure_client()
        area = build_area(lat, lon, 0.5, 0.5)

        today = date.today()

        times_all = []
        values_all = {}  # species -> list
        meta_notes = {"segments": []}

        # ---------- 1) Историческая часть: от d0 до min(d1, today) включительно ----------
        hist_end = min(d1, today)
        cur = d0
        while cur <= hist_end:
            date_str = cur.isoformat()
            leads = [str(h) for h in (0,3,6,9,12,15,18,21,24)]
            req = {"model":["ensemble"], "date":f"{date_str}/{date_str}", "format":"netcdf",
                   "type":["forecast"], "time":["00:00"], "leadtime_hour":leads, "level":["0"], "area": area}

            # primary
            t1 = cache / f"cams_{date_str}_series_{lat:.2f}_{lon:.2f}.nc"
            mode1, path1 = retrieve(client, PRIMARY_DATASET, req, t1, EXTENDED_VARS)
            times1, vals1, meta1 = extract_series_hourly_for_hours(path1, lat, lon, date_str, 24)

            # pollen
            t2 = cache / f"pollen_{date_str}_series_{lat:.2f}_{lon:.2f}.nc"
            try:
                mode2, path2 = retrieve(client, POLLEN_DATASET, req, t2, EXTENDED_VARS)
                times2, vals2, meta2 = extract_series_hourly_for_hours(path2, lat, lon, date_str, 24)
            except Exception:
                times2, vals2, meta2, mode2 = [], {}, {}, "none"

            times, values, _ = merge_series(times1, vals1, meta1, times2, vals2, meta2)

            # склейка (без двойного полуночи): первый день кладём 25 точек, остальные — 24 (с 01:00)
            if not times_all:
                times_all.extend(times)
                for sp, arr in values.items():
                    values_all.setdefault(sp, []).extend(arr)
            else:
                times_all.extend(times[1:])
                for sp, arr in values.items():
                    values_all.setdefault(sp, []).extend(arr[1:])

            meta_notes["segments"].append({"date": date_str, "mode": "historical"})
            cur += timedelta(days=1)

        # ---------- 2) Прогнозная часть: если d1 > today ----------
        if d1 > today:
            # старт прогноза — max(today, d0)
            base = max(today, d0)
            end_forecast = min(d1, base + timedelta(days=4))
            total_hours = int((end_forecast - base).days * 24 + 24)
            total_hours = min(total_hours, FORECAST_MAX_H)

            base_str = base.isoformat()
            leads = [str(h) for h in range(0, total_hours+1, 3)]
            req = {"model":["ensemble"], "date":f"{base_str}/{base_str}", "format":"netcdf",
                   "type":["forecast"], "time":["00:00"], "leadtime_hour":leads, "level":["0"], "area": area}

            # primary
            t1 = cache / f"cams_{base_str}_H{total_hours}_{lat:.2f}_{lon:.2f}.nc"
            mode1, path1 = retrieve(client, PRIMARY_DATASET, req, t1, EXTENDED_VARS)
            times1, vals1, meta1 = extract_series_hourly_for_hours(path1, lat, lon, base_str, total_hours)

            # pollen
            t2 = cache / f"pollen_{base_str}_H{total_hours}_{lat:.2f}_{lon:.2f}.nc"
            try:
                mode2, path2 = retrieve(client, POLLEN_DATASET, req, t2, EXTENDED_VARS)
                times2, vals2, meta2 = extract_series_hourly_for_hours(path2, lat, lon, base_str, total_hours)
            except Exception:
                times2, vals2, meta2, mode2 = [], {}, {}, "none"

            times, values, _ = merge_series(times1, vals1, meta1, times2, vals2, meta2)

            # склейка с историей: убираем дубли времени
            if times_all:
                # если первая точка прогноза совпадает с последней исторической — выкинем первую
                if times[0] == times_all[-1]:
                    times = times[1:]
                    for sp in values.keys():
                        values[sp] = values[sp][1:]

            times_all.extend(times)
            for sp, arr in values.items():
                values_all.setdefault(sp, []).extend(arr)

            meta_notes["segments"].append({"date": base_str, "mode": "forecast", "hours": total_hours})

        # приведение длины по всем видам к одному размеру
        max_len = len(times_all)
        for sp, arr in list(values_all.items()):
            if len(arr) < max_len:
                values_all[sp] = arr + [None]*(max_len-len(arr))

        loc = {"lat": float(lat), "lon": float(lon)}
        if loc_name: loc["name"] = loc_name

        return jsonify({
            "status":"success","location":loc,
            "times": times_all, "unit": UNIT, "values": values_all,
            "metadata": {
                "note":"range built from daily slices + forecast up to 4 days if needed",
                **meta_notes
            }
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

# ---------- очистка кеша ----------
@app.post("/api/clear_cache")
def clear_cache():
    try:
        cache = pathlib.Path("cache")
        if cache.exists(): shutil.rmtree(cache)
        cache.mkdir(exist_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ------------------ run ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
