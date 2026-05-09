"""
NightWalk — Manual Map Matcher
=====================================
Usage:
    python map_matcher.py night-photos/ urban-mosaic/washington-square.csv --output matches_remapped.csv
"""

import sys, math, argparse, json
from pathlib import Path
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView

# ── Geo & Logic Helpers ───────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_candidates(lat, lon, df, n=4):
    actual_pre_filter = max(20, n * 2)
    df = df.copy()
    df["_dist"] = df.apply(lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1)
    return df.nsmallest(actual_pre_filter, "_dist").head(n).reset_index(drop=True)

# ── UI Components ─────────────────────────────────────────────────────────────

CARD_W = 260
CARD_H = 240
IMG_H  = 160

class CandidateCard(QWidget):
    def __init__(self, row, image_root, on_select, used_day_ids):
        super().__init__()
        self._on_select = on_select
        self._row = row
        self.selected = False
        self.is_used = int(float(row["id"])) in used_day_ids

        self.setFixedSize(CARD_W, CARD_H)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.img_label = QLabel()
        self.img_label.setFixedSize(CARD_W - 12, IMG_H)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background: #111; border-radius: 4px;")

        img_path = image_root / str(row["image"]).strip()
        if img_path.exists():
            pix = QPixmap(str(img_path)).scaled(CARD_W - 12, IMG_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(pix)
        else:
            self.img_label.setText("image\nnot found")
            self.img_label.setStyleSheet("background:#111; color:#555; border-radius:4px;")

        layout.addWidget(self.img_label)

        dist_m = float(row.get("_dist", 0))
        heading = float(row.get("heading", row.get("azimuth", 0)))

        meta_html = (
            f"<b>{dist_m:.0f}m from pin</b> &nbsp;·&nbsp; {heading:.0f}° heading<br>"
            f"<span style='color:#888;font-size:11px'>ID {row['id']}</span>"
        )

        if self.is_used:
            meta_html += "<br><span style='color:#ff4757; font-weight:bold;'>⚠️ ALREADY MATCHED</span>"

        meta = QLabel(meta_html)
        meta.setTextFormat(Qt.RichText)
        meta.setWordWrap(True)
        meta.setStyleSheet("font-size: 12px;")
        layout.addWidget(meta)

        self._update_style()

    def _update_style(self):
        if self.selected:
            self.setStyleSheet("CandidateCard { background: #0a3d62; border: 2px solid #378ADD; border-radius: 8px; }")
        elif self.is_used:
            self.setStyleSheet("CandidateCard { background: #2a1111; border: 2px solid #552222; border-radius: 8px; }"
                               "CandidateCard:hover { border: 2px solid #ff4757; background: #3a1515; }")
        else:
            self.setStyleSheet("CandidateCard { background: #1e1e24; border: 2px solid #333; border-radius: 8px; }"
                               "CandidateCard:hover { border: 2px solid #555; background: #26262e; }")

    def mousePressEvent(self, event):
        self._on_select(self._row)

    def set_selected(self, val):
        self.selected = val
        self._update_style()


class MapMatcherWindow(QMainWindow):
    def __init__(self, pending_photos, df, image_root, output_path, n_candidates, used_day_ids):
        super().__init__()
        self.pending_photos = pending_photos
        self.df = df
        self.image_root = image_root
        self.output_path = output_path
        self.n_candidates = n_candidates
        self.used_day_ids = used_day_ids
        self.current_idx = 0

        self._cards = []
        self._selected_row = None
        self._current_map_lat = None
        self._current_map_lon = None
        self._map_ready = False

        self.setWindowTitle("NightWalk — Manual Map Matcher")
        self.resize(1800, 900)
        self._build_ui()
        self._load_current()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)

        # ── Left: Night Photo ──
        left = QVBoxLayout()
        self.progress_label = QLabel()
        self.progress_label.setFont(QFont("", 12))
        left.addWidget(self.progress_label)

        self.night_img = QLabel()
        self.night_img.setFixedSize(450, 450)
        self.night_img.setAlignment(Qt.AlignCenter)
        self.night_img.setStyleSheet("background:#0d0d0d; border-radius:8px;")
        left.addWidget(self.night_img)

        self.night_meta = QLabel()
        self.night_meta.setStyleSheet("font-size:12px; color:#aaa;")
        left.addWidget(self.night_meta)

        btn_row = QHBoxLayout()
        self.skip_btn = QPushButton("Skip / Can't find")
        self.skip_btn.setFixedHeight(38)
        self.skip_btn.clicked.connect(self._skip)

        self.confirm_btn = QPushButton("✓ Confirm match")
        self.confirm_btn.setFixedHeight(38)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._confirm)
        self.confirm_btn.setStyleSheet(
            "QPushButton { background:#0a3d62; color:#E6F1FB; border-radius:6px; font-size:13px; }"
            "QPushButton:disabled { background:#222; color:#555; }"
        )
        btn_row.addWidget(self.skip_btn)
        btn_row.addWidget(self.confirm_btn)
        left.addLayout(btn_row)
        left.addStretch()
        main.addLayout(left, stretch=2)

        # ── Center: Interactive Map ──
        center = QVBoxLayout()
        map_title = QLabel("Map Pin (Red = Original GPS)")
        map_title.setFont(QFont("", 12, QFont.Bold))
        center.addWidget(map_title)

        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._load_map()
        self.web_view.titleChanged.connect(self._on_map_clicked)
        self.web_view.loadFinished.connect(self._on_map_loaded)

        center.addWidget(self.web_view, stretch=1)
        main.addLayout(center, stretch=3)

        # ── Right: Candidates ──
        right = QVBoxLayout()
        right_header = QHBoxLayout()
        cand_title = QLabel("Candidates from search pin:")
        cand_title.setFont(QFont("", 12, QFont.Bold))
        right_header.addWidget(cand_title)
        right_header.addStretch()

        right_header.addWidget(QLabel("Show:"))
        self.cand_spinbox = QSpinBox()
        self.cand_spinbox.setRange(1, 200)
        self.cand_spinbox.setValue(self.n_candidates)
        self.cand_spinbox.valueChanged.connect(self._on_spin_changed)
        self.cand_spinbox.setStyleSheet("background:#333; color:#fff; padding:4px;")
        right_header.addWidget(self.cand_spinbox)

        right.addLayout(right_header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_widget)

        right.addWidget(self.scroll)  # ✅ THE FIX: scroll area was never added to the layout

        main.addLayout(right, stretch=4)

    def _on_spin_changed(self, value):
        self.n_candidates = value
        if self._current_map_lat is not None and self._current_map_lon is not None:
            self._search_candidates()

    def _load_map(self):
        avg_lat = self.df["lat"].mean()
        avg_lon = self.df["lon"].mean()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style> body, html, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; background:#222; }} </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([{avg_lat}, {avg_lon}], 16);
                L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                    maxZoom: 19,
                    attribution: '&copy; CartoDB'
                }}).addTo(map);

                var clickMarker = null;
                var suggestMarker = null;
                var candidateLayer = L.layerGroup().addTo(map);

                function setSuggestedPin(lat, lon) {{
                    if (suggestMarker) map.removeLayer(suggestMarker);
                    suggestMarker = L.circleMarker([lat, lon], {{
                        color: '#ff4757',
                        fillColor: '#ff4757',
                        fillOpacity: 0.8,
                        radius: 7
                    }}).addTo(map);

                    if (clickMarker) map.removeLayer(clickMarker);
                    map.setView([lat, lon], 17);
                }}

                function drawCandidates(candidates) {{
                    candidateLayer.clearLayers();
                    candidates.forEach(function(c) {{
                        var color = c.used ? '#ff4757' : '#378ADD';
                        var m = L.circleMarker([c.lat, c.lon], {{
                            color: color,
                            fillColor: color,
                            fillOpacity: 0.8,
                            radius: 5
                        }});
                        m.bindTooltip("ID: " + c.id + "<br>Dist: " + c.dist + "m");
                        candidateLayer.addLayer(m);
                    }});
                }}

                map.on('click', function(e) {{
                    if (clickMarker) map.removeLayer(clickMarker);
                    clickMarker = L.marker(e.latlng).addTo(map);
                    document.title = "MAP_CLICK:" + e.latlng.lat + "," + e.latlng.lng;
                }});
            </script>
        </body>
        </html>
        """
        self.web_view.setHtml(html)

    def _on_map_loaded(self, ok):
        if ok:
            self._map_ready = True
            self._update_map_suggestion()

    def _on_map_clicked(self, title):
        if title.startswith("MAP_CLICK:"):
            _, coords = title.split(":")
            lat_str, lon_str = coords.split(",")
            self._current_map_lat = float(lat_str)
            self._current_map_lon = float(lon_str)
            self._search_candidates()

    def _update_map_suggestion(self):
        if not self._map_ready or self.current_idx >= len(self.pending_photos):
            return

        data = self.pending_photos[self.current_idx]
        orig_lat = data.get("lat")
        orig_lon = data.get("lon")

        try:
            if pd.notna(orig_lat) and pd.notna(orig_lon) and str(orig_lat).strip() != "":
                lat_f = float(orig_lat)
                lon_f = float(orig_lon)
                self.web_view.page().runJavaScript(f"setSuggestedPin({lat_f}, {lon_f});")

                # Auto-load candidates from the suggested pin
                self._current_map_lat = lat_f
                self._current_map_lon = lon_f
                self._search_candidates()

        except Exception as e:
            print(f"Could not set suggested pin: {e}")

    def _load_current(self):
        if self.current_idx >= len(self.pending_photos):
            self.night_img.setText("All done! No more unmapped photos.")
            self.night_meta.clear()
            self.progress_label.setText("Finished")
            self.confirm_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            return

        data = self.pending_photos[self.current_idx]
        path = data["path"]

        self.progress_label.setText(f"Photo {self.current_idx + 1} of {len(self.pending_photos)}")
        self.night_meta.setText(f"<b>{path.name}</b>")

        pix = QPixmap(str(path))
        if not pix.isNull():
            self.night_img.setPixmap(pix.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.night_img.setText("Cannot load image")

        self._current_map_lat = None
        self._current_map_lon = None
        self._selected_row = None
        self.confirm_btn.setEnabled(False)

        # Nuke and rebuild the scroll container
        if self.scroll.widget():
            self.scroll.widget().deleteLater()

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_widget)

        self._cards.clear()

        # Trigger the suggested pin logic (which also loads candidates)
        self._update_map_suggestion()

    def _search_candidates(self):
        if self._current_map_lat is None or self._current_map_lon is None:
            return

        candidates = find_candidates(self._current_map_lat, self._current_map_lon, self.df, self.n_candidates)

        # Nuke and rebuild the scroll container
        if self.scroll.widget():
            self.scroll.widget().deleteLater()

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_widget)

        self._cards.clear()
        self._selected_row = None
        self.confirm_btn.setEnabled(False)

        cands_for_map = []

        for _, row in candidates.iterrows():
            day_id = int(float(row["id"]))

            cands_for_map.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "id": day_id,
                "dist": round(float(row["_dist"])),
                "used": day_id in self.used_day_ids
            })

            card = CandidateCard(row, self.image_root, self._on_card_selected, self.used_day_ids)
            self._cards.append(card)

        # Safe column calculation — guard against zero-width viewport on first render
        viewport_w = self.scroll.viewport().width()
        if viewport_w <= 0:
            viewport_w = CARD_W + 12
        cols = max(1, viewport_w // (CARD_W + 12))

        for i, card in enumerate(self._cards):
            self.grid_layout.addWidget(card, i // cols, i % cols)

        js_code = f"drawCandidates({json.dumps(cands_for_map)});"
        self.web_view.page().runJavaScript(js_code)

    def _on_card_selected(self, row):
        self._selected_row = row
        for card in self._cards:
            card.set_selected(card._row["id"] == row["id"])
        self.confirm_btn.setEnabled(True)

    def _confirm(self):
        if self._selected_row is None:
            return

        path = self.pending_photos[self.current_idx]["path"]
        row = self._selected_row

        day_id = int(float(row["id"]))

        updates = {
            "day_image":    str(row["image"]).strip(),
            "day_id":       day_id,
            "day_lat":      float(row["lat"]),
            "day_lon":      float(row["lon"]),
            "day_heading":  float(row.get("heading", row.get("azimuth", 0))),
            "distance_m":   round(float(row["_dist"]), 2),
            "skipped":      False,
        }

        self.used_day_ids.add(day_id)
        self._update_csv_row(path.name, updates)

    def _skip(self):
        self.current_idx += 1
        self._load_current()

    def _update_csv_row(self, filename, updates):
        df_out = pd.read_csv(self.output_path)
        idx = df_out.index[df_out["night_photo"] == filename].tolist()

        if idx:
            row_idx = idx[0]
            for col, val in updates.items():
                df_out[col] = df_out[col].astype(object)
                df_out.at[row_idx, col] = val
            df_out.to_csv(self.output_path, index=False)
        else:
            print(f"Warning: {filename} was not found in the CSV. Could not update.")

        self.current_idx += 1
        self._load_current()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("night_dir", help="Folder of night photos")
    parser.add_argument("csv",       help="Daytime image CSV")
    parser.add_argument("--image-root",  default=None)
    parser.add_argument("--output",      default="matches.csv")
    parser.add_argument("--candidates",  type=int, default=12)
    parser.add_argument("--mode",        choices=["auto", "continue", "skipped"], default="auto",
                        help="auto: smart default | continue: unlabeled only | skipped: fix skipped only")
    args = parser.parse_args()

    night_dir  = Path(args.night_dir)
    csv_path   = Path(args.csv).expanduser().absolute()
    image_root = Path(args.image_root).expanduser().absolute() if args.image_root else csv_path.parent / csv_path.stem
    output_path = Path(args.output)

    extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".heic", ".HEIC"}

    # ── Bootstrap output CSV if it doesn't exist yet ──────────────────────────
    if not output_path.exists():
        photos = sorted([p for p in night_dir.iterdir() if p.suffix in extensions])
        if not photos:
            print(f"Error: No photos found in {night_dir}")
            sys.exit(1)
        n = len(photos)
        bootstrap_df = pd.DataFrame({
            "night_photo": [p.name for p in photos],
            "night_lat":   [None] * n,
            "night_lon":   [None] * n,
            "day_image":   [None] * n,
            "day_id":      [None] * n,
            "day_lat":     [None] * n,
            "day_lon":     [None] * n,
            "day_heading": [None] * n,
            "distance_m":  [None] * n,
            "skipped":     [None] * n,   # None = unlabeled, True = skipped, False = matched
        })
        bootstrap_df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {n} photos. Starting from scratch.")

    existing_df = pd.read_csv(output_path)

    # Classify rows
    matched_mask   = (existing_df["skipped"] == False) & existing_df["day_id"].notna()
    skipped_mask   = existing_df["skipped"] == True
    unlabeled_mask = existing_df["skipped"].isna()

    used_day_ids = set(existing_df.loc[matched_mask, "day_id"].astype(float).astype(int))

    # ── Determine which rows to show based on mode ────────────────────────────
    mode = args.mode

    # "auto": first run = continue (unlabeled); if all labeled = skipped
    if mode == "auto":
        if unlabeled_mask.any():
            mode = "continue"
        elif skipped_mask.any():
            mode = "skipped"
        else:
            print("All photos are matched! Nothing to do.")
            sys.exit(0)

    if mode == "continue":
        work_df = existing_df[unlabeled_mask]
        mode_label = "unlabeled"
    elif mode == "skipped":
        work_df = existing_df[skipped_mask]
        mode_label = "skipped"

    pending_photos = []
    for _, row in work_df.iterrows():
        fname = row["night_photo"]
        p = night_dir / fname
        if p.exists() and p.suffix in extensions:
            pending_photos.append({
                "path": p,
                "lat":  row.get("night_lat", None),
                "lon":  row.get("night_lon", None),
            })

    print(f"\nMode: {mode} — found {len(pending_photos)} {mode_label} photos.")
    if not pending_photos:
        print(f"No {mode_label} photos to label. Try a different --mode.")
        sys.exit(0)

    # ── Load daytime CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    for col in ["lat", "lon", "snapped_lat", "snapped_lon", "heading", "azimuth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "snapped_lat" in df.columns: df["lat"] = df["snapped_lat"].fillna(df["lat"])
    if "snapped_lon" in df.columns: df["lon"] = df["snapped_lon"].fillna(df["lon"])
    if "heading" not in df.columns and "azimuth" in df.columns: df["heading"] = df["azimuth"]
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,     QColor(22, 22, 28))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base,       QColor(15, 15, 20))
    palette.setColor(QPalette.Text,       QColor(220, 220, 220))
    app.setPalette(palette)

    win = MapMatcherWindow(pending_photos, df, image_root, output_path, args.candidates, used_day_ids)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()