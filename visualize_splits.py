"""visualize_splits.py
Plots every sampled image as a dot, colored by grid cell (shuffled so
adjacent cells get distinct colors). Click any dot to highlight that
cell with a bounding rectangle and see its images.

Usage:
  1. python -m http.server 8000   (from project root)
  2. python visualize_splits.py
  3. open http://localhost:8000/split_map.html
"""

import pandas as pd
import json
import random
from pathlib import Path

SPLIT_CSV          = Path("splits/efficientnet_train_images.csv")
IMAGE_ROOT         = "urban-mosaic/washington-square"
OUT_HTML           = Path("split_map.html")
MAX_PREVIEW_IMAGES = 6
LAT_BIN_SIZE       = 0.0005
LON_BIN_SIZE       = 0.0005

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(SPLIT_CSV)

# Shuffle cell order before assigning hues so spatial neighbors differ
unique_cells = sorted(df["grid_cell"].unique())
shuffled = unique_cells.copy()
random.seed(99)
random.shuffle(shuffled)
n_cells = len(shuffled)
cell_to_idx = {cell: i for i, cell in enumerate(shuffled)}
df["cell_idx"] = df["grid_cell"].map(cell_to_idx)

# Per-cell metadata + bounding box
cell_meta = {}
cell_images_map = {}
for cell_id, group in df.groupby("grid_cell"):
    lat_c = group["snapped_lat"].mean()
    lon_c = group["snapped_lon"].mean()
    cell_meta[cell_id] = {
        "count":         len(group),
        "neighbourhood": group["neighbourhood"].mode()[0],
        "n_vehicles":    group["android_id"].nunique(),
        "periods":       ", ".join(sorted(group["period"].unique())),
        "cell_idx":      int(cell_to_idx[cell_id]),
        # bounding box corners in [lat, lon] for L.rectangle
        "bbox": [
            [lat_c - LAT_BIN_SIZE / 2, lon_c - LON_BIN_SIZE / 2],
            [lat_c + LAT_BIN_SIZE / 2, lon_c + LON_BIN_SIZE / 2],
        ],
    }
    cell_images_map[cell_id] = group["image"].sample(
        min(len(group), MAX_PREVIEW_IMAGES), random_state=42
    ).tolist()

# All image points
points = df[["snapped_lat", "snapped_lon", "grid_cell", "cell_idx"]].rename(
    columns={"snapped_lat": "lat", "snapped_lon": "lon"}
).to_dict("records")

stats = {
    "total_images":   len(df),
    "total_cells":    n_cells,
    "total_vehicles": df["android_id"].nunique(),
}

print(f"{len(points):,} points, {n_cells} cells")

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Split map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f5f5f3; color: #1a1a1a;
        display: flex; flex-direction: column; height: 100vh; padding: 16px; gap: 12px; }}
h1 {{ font-size: 16px; font-weight: 500; flex-shrink: 0; }}
.stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; flex-shrink: 0; }}
.stat {{ background: #fff; border: 0.5px solid #ddd; border-radius: 10px; padding: 8px 12px; }}
.stat-label {{ font-size: 11px; color: #888; }}
.stat-value {{ font-size: 20px; font-weight: 500; }}
.main {{ display: flex; gap: 12px; flex: 1; min-height: 0; }}
#map {{ flex: 1; border-radius: 10px; border: 0.5px solid #ddd; }}
.panel {{ width: 280px; flex-shrink: 0; display: flex; flex-direction: column; gap: 10px; overflow: hidden; }}
.info-box {{ background: #fff; border: 0.5px solid #ddd; border-radius: 10px;
             padding: 12px; font-size: 12px; color: #444; line-height: 1.9; flex-shrink: 0; }}
.info-title {{ font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #1a1a1a;
               display: flex; align-items: center; gap: 8px; }}
.swatch {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
#img-panel {{ background: #fff; border: 0.5px solid #ddd; border-radius: 10px;
              padding: 12px; flex: 1; overflow-y: auto; }}
.img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }}
.img-grid img {{ width: 100%; aspect-ratio: 4/3; object-fit: cover;
                 border-radius: 6px; border: 0.5px solid #ddd;
                 cursor: pointer; transition: opacity 0.15s; }}
.img-grid img:hover {{ opacity: 0.8; }}
.placeholder {{ color: #aaa; font-size: 12px; font-style: italic; }}
#lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.82);
             z-index: 9999; align-items: center; justify-content: center; }}
#lightbox.open {{ display: flex; }}
#lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 8px; }}
#lightbox-close {{ position: fixed; top: 18px; right: 22px; font-size: 28px;
                   color: #fff; cursor: pointer; user-select: none; }}
</style>
</head>
<body>

<h1>Split map — {SPLIT_CSV.stem}</h1>

<div class="stats">
  <div class="stat">
    <div class="stat-label">sampled images</div>
    <div class="stat-value">{stats["total_images"]:,}</div>
  </div>
  <div class="stat">
    <div class="stat-label">grid cells</div>
    <div class="stat-value">{stats["total_cells"]:,}</div>
  </div>
  <div class="stat">
    <div class="stat-label">vehicle routes</div>
    <div class="stat-value">{stats["total_vehicles"]:,}</div>
  </div>
</div>

<div class="main">
  <div id="map"></div>
  <div class="panel">
    <div class="info-box" id="info">
      <div class="placeholder">Click any dot to see cell details.</div>
    </div>
    <div id="img-panel">
      <div class="placeholder">Sample images will appear here.</div>
    </div>
  </div>
</div>

<div id="lightbox" onclick="closeLightbox()">
  <span id="lightbox-close">&#x2715;</span>
  <img id="lightbox-img" src="" alt="full size">
</div>

<script>
const IMAGE_ROOT = "{IMAGE_ROOT}";
const N_CELLS    = {n_cells};
const points     = {json.dumps(points)};
const cellMeta   = {json.dumps(cell_meta)};
const cellImages = {json.dumps(cell_images_map)};

const map = L.map("map").setView([40.7295, -73.998], 15);
L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
  attribution: "© OpenStreetMap © CARTO",
  subdomains: "abcd", maxZoom: 19
}}).addTo(map);

function cellColor(idx) {{
  const hue = (idx * (360 / N_CELLS)) % 360;
  return `hsl(${{hue}}, 72%, 45%)`;
}}

let activeRect = null;
let activeCell = null;

function showPanel(cellId) {{
  if (activeCell === cellId) return;
  activeCell = cellId;

  const meta  = cellMeta[cellId];
  const imgs  = cellImages[cellId] || [];
  const color = cellColor(meta.cell_idx);

  // Draw bounding box
  if (activeRect) map.removeLayer(activeRect);
  activeRect = L.rectangle(meta.bbox, {{
    color: color, weight: 2.5, fillColor: color,
    fillOpacity: 0.12, dashArray: "4 3"
  }}).addTo(map);

  document.getElementById("info").innerHTML = `
    <div class="info-title">
      <span class="swatch" style="background:${{color}}"></span>
      ${{meta.neighbourhood}}
    </div>
    <b>Images in cell:</b> ${{meta.count}}<br>
    <b>Vehicles:</b> ${{meta.n_vehicles}}<br>
    <b>Periods:</b> ${{meta.periods}}
  `;

  const imgPanel = document.getElementById("img-panel");
  if (!imgs.length) {{
    imgPanel.innerHTML = '<div class="placeholder">No images.</div>';
    return;
  }}
  imgPanel.innerHTML =
    `<div style="font-size:13px;font-weight:500;margin-bottom:8px">
       ${{imgs.length}} sample image${{imgs.length !== 1 ? "s" : ""}}</div>
     <div class="img-grid">${{
       imgs.map(p => {{
         const src = IMAGE_ROOT + "/" + p;
         return `<img src="${{src}}" loading="lazy"
                      onerror="this.style.display='none'"
                      onclick="openLightbox('${{src}}')" />`;
       }}).join("")
     }}</div>`;
}}

points.forEach(pt => {{
  const color = cellColor(pt.cell_idx);
  L.circleMarker([pt.lat, pt.lon], {{
    radius: 4, fillColor: color,
    color: "rgba(255,255,255,0.5)", weight: 0.5, fillOpacity: 0.9
  }}).addTo(map).on("click", () => showPanel(pt.grid_cell));
}});

function openLightbox(src) {{
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.add("open");
}}
function closeLightbox() {{
  document.getElementById("lightbox").classList.remove("open");
  document.getElementById("lightbox-img").src = "";
}}
document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeLightbox(); }});
</script>
</body>
</html>
"""

OUT_HTML.write_text(html)
print(f"Saved → {OUT_HTML}")
print(f"\nTo view:")
print(f"  1. python -m http.server 8000   (from project root)")
print(f"  2. open http://localhost:8000/split_map.html")