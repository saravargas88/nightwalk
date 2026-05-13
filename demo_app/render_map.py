"""render_map.py — folium HTML maps showing shortest vs well-lit routes.

Two ways to drive it:

    # one route between explicit lat/lon points (single map):
    python demo_app/render_map.py --start 40.7295,-73.9970 --end 40.7340,-73.9920

    # gallery of canned start/end pairs for §5.2 figure (two HTMLs by default):
    python demo_app/render_map.py --examples

The default --layout is `both`, producing:
  • route_viewer_grid.html    — three side-by-side mini-maps (report figure)
  • route_viewer_layered.html — single map with one toggleable layer per pair
"""

from __future__ import annotations

import argparse
from pathlib import Path

from route import (RouteResult, compute_routes, load_edge_scores,
                   load_edge_sources, load_edge_table, summary_lines)

DEMO_ROOT = Path(__file__).resolve().parent
DEMO_DATA = DEMO_ROOT / "data"
DEFAULT_GRAPH = DEMO_DATA / "wsp_walk.graphml"
DEFAULT_SCORES = DEMO_DATA / "edge_lighting.csv"
DEFAULT_OUT = DEMO_ROOT / "route_viewer.html"
DEFAULT_GRID_OUT = DEMO_ROOT / "route_viewer_grid.html"
DEFAULT_LAYERED_OUT = DEMO_ROOT / "route_viewer_layered.html"

# Hand-picked start/end pairs around WSP for the qualitative gallery (§5.2).
EXAMPLE_PAIRS = [
    ("WSP arch → NYU Stern", (40.7308, -73.9973), (40.7295, -73.9930)),
    ("West Village → Astor Pl", (40.7330, -73.9990), (40.7295, -73.9910)),
    ("Bleecker St → Union Sq", (40.7280, -73.9970), (40.7355, -73.9910)),
]

# Per-pair accent colours for marker pins in the layered/grid views — keeps
# pairs distinguishable when shown together.
PAIR_ACCENTS = ["#1f77b4", "#d62728", "#2ca02c"]

DARK_TILE = "cartodbdark_matter"


def _color_for(score: float) -> str:
    """Navy (dark) → midnight blue → amber → bright yellow (bright). score ∈ [0, 1]."""
    score = max(0.0, min(1.0, score))
    navy   = (0x0b, 0x1d, 0x51)
    midb   = (0x1f, 0x3a, 0x93)
    amber  = (0xf5, 0xb8, 0x00)
    yellow = (0xff, 0xe3, 0x4d)

    def lerp(a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    if score < 0.33:
        r, g, b = lerp(navy, midb, score / 0.33)
    elif score < 0.66:
        r, g, b = lerp(midb, amber, (score - 0.33) / 0.33)
    else:
        r, g, b = lerp(amber, yellow, (score - 0.66) / 0.34)
    return f"#{r:02x}{g:02x}{b:02x}"


ROUTE_SHORTEST_COLOR = "#3388ff"   # blue
ROUTE_LIT_COLOR      = "#ff2d95"   # hot pink
ROUTE_LIT_DASH       = "8,6"
FALLBACK_GRAY        = "#9aa0a6"


# ── building blocks ────────────────────────────────────────────────────────
def _add_basemap(fmap, G, edge_scores, edge_sources: dict | None = None,
                 layer=None):
    """Colour every edge by its lighting_score. Renders into `layer` if given,
    else directly into `fmap`."""
    import folium

    target = layer if layer is not None else fmap

    for u, v, k, data in G.edges(keys=True, data=True):
        edge_key = (u, v, k)
        score = edge_scores.get(edge_key, 0.5)
        source = edge_sources.get(edge_key) if edge_sources else None
        if source == "fallback":
            color = FALLBACK_GRAY
            opacity = 0.30
        else:
            color = _color_for(score)
            opacity = 0.70  # bumped up for dark-tile contrast
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G.nodes[u]["y"], G.nodes[u]["x"]),
                      (G.nodes[v]["y"], G.nodes[v]["x"])]
        folium.PolyLine(coords, color=color, weight=2.5, opacity=opacity).add_to(target)


def _add_route(target, route: RouteResult, color: str, weight: int = 7,
               dash_array: str | None = None):
    import folium

    kwargs = dict(color=color, weight=weight, opacity=0.95,
                  tooltip=f"{route.label}: {route.distance_m:.0f} m, mean lighting {route.mean_lighting:.2f}")
    if dash_array:
        kwargs["dash_array"] = dash_array
    folium.PolyLine(route.coords, **kwargs).add_to(target)


def _add_markers(target, start, end, title, accent: str | None = None):
    import folium

    folium.CircleMarker(start, radius=8, color=accent or "#1f77b4",
                        fill=True, fill_opacity=0.95,
                        tooltip=f"{title} — start").add_to(target)
    folium.CircleMarker(end, radius=8, color=accent or "#1f77b4",
                        fill=True, fill_opacity=0.95, weight=3,
                        tooltip=f"{title} — end").add_to(target)


# ── small UI fragments ─────────────────────────────────────────────────────
def _title_strip_html(mode: str, metric: str, lambda_: float,
                      unknown_penalty: float, n_pairs: int) -> str:
    metric_label = "median (typical)" if metric == "median" else "p25 (worst quartile)"
    gradient_css = (
        "background: linear-gradient(to right, "
        "#0b1d51 0%, #1f3a93 33%, #f5b800 66%, #ffe34d 100%);"
    )
    return f"""
    <div style="
      background:#11151c; color:#e6e9ef; padding:14px 18px;
      font: 14px/1.4 system-ui, -apple-system, sans-serif;
      border-bottom:1px solid #1f2530; box-shadow:0 1px 4px rgba(0,0,0,0.25);">
      <div style="font-size:18px; font-weight:700; margin-bottom:2px;">
        NightWalk · Well-lit vs Shortest Pedestrian Routes
      </div>
      <div style="font-size:12px; color:#9aa6b8; margin-bottom:10px;">
        aggregation = <b>{mode}</b> · metric = <b>{metric_label}</b> · λ = <b>{lambda_}</b> · μ_unk = <b>{unknown_penalty}</b> · {n_pairs} example pair{'s' if n_pairs != 1 else ''}
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:12px;">
        <span style="display:inline-flex; align-items:center;">
          <span style="display:inline-block; width:28px; height:4px; background:{ROUTE_SHORTEST_COLOR}; margin-right:6px;"></span>
          shortest path (distance-only)
        </span>
        <span style="display:inline-flex; align-items:center;">
          <span style="display:inline-block; width:28px; height:4px;
                       background:repeating-linear-gradient(to right,{ROUTE_LIT_COLOR} 0 6px,transparent 6px 10px);
                       margin-right:6px;"></span>
          well-lit path
        </span>
        <span style="display:inline-flex; align-items:center;">
          <span style="display:inline-block; width:120px; height:10px; {gradient_css}
                       border:1px solid #4a5260; margin:0 8px;"></span>
          dark → bright
        </span>
        <span style="display:inline-flex; align-items:center;">
          <span style="display:inline-block; width:14px; height:10px; background:{FALLBACK_GRAY}; border:1px solid #4a5260; margin-right:6px;"></span>
          no direct evidence
        </span>
      </div>
    </div>
    """


def _panel_caption_html(title: str, short: RouteResult, lit: RouteResult,
                        accent: str) -> str:
    detour = (lit.distance_m / short.distance_m) if short.distance_m else 1.0
    light_delta = lit.mean_lighting - short.mean_lighting
    delta_sign = "+" if light_delta >= 0 else ""
    return f"""
    <div style="padding:10px 14px; background:#181d27; color:#e6e9ef;
                border-top:3px solid {accent}; font:13px/1.4 system-ui, sans-serif;">
      <div style="font-size:15px; font-weight:700; margin-bottom:6px;">{title}</div>
      <div style="display:flex; gap:18px; flex-wrap:wrap;">
        <div>
          <div style="font-size:11px; color:#7e8a9c; text-transform:uppercase; letter-spacing:0.04em;">Shortest</div>
          <div><b>{short.distance_m:.0f} m</b>  ·  light <b>{short.mean_lighting:.2f}</b></div>
        </div>
        <div>
          <div style="font-size:11px; color:#7e8a9c; text-transform:uppercase; letter-spacing:0.04em;">Well-lit</div>
          <div><b>{lit.distance_m:.0f} m</b>  ·  light <b>{lit.mean_lighting:.2f}</b></div>
        </div>
        <div>
          <div style="font-size:11px; color:#7e8a9c; text-transform:uppercase; letter-spacing:0.04em;">Detour</div>
          <div><b>{detour:.2f}×</b>  ·  Δ light <b>{delta_sign}{light_delta:.2f}</b></div>
        </div>
      </div>
    </div>
    """


def _overall_table_html(items: list[tuple[str, RouteResult, RouteResult]]) -> str:
    rows = []
    for title, short, lit in items:
        rows.append(
            f"<tr><td><b>{title}</b></td>"
            f"<td>{short.distance_m:.0f}</td><td>{short.mean_lighting:.2f}</td>"
            f"<td>{lit.distance_m:.0f}</td><td>{lit.mean_lighting:.2f}</td>"
            f"<td>{(lit.distance_m/short.distance_m if short.distance_m else 1):.2f}×</td></tr>"
        )
    body = "".join(rows)
    return f"""
    <div style="position:fixed; bottom:14px; left:14px; z-index:9999;
                background:rgba(17,21,28,0.92); color:#e6e9ef; padding:10px 14px;
                border:1px solid #2a3140; border-radius:8px;
                font:12px/1.4 system-ui, sans-serif; box-shadow:0 1px 6px rgba(0,0,0,0.4);">
      <table style="border-collapse:collapse;">
        <tr style="text-align:left; border-bottom:1px solid #2a3140;">
          <th style="padding-right:14px;">pair</th><th colspan="2">shortest</th><th colspan="2">well-lit</th><th>detour</th>
        </tr>
        <tr style="font-size:11px; color:#7e8a9c;">
          <th></th><th style="padding-right:8px;">dist (m)</th><th>light</th>
          <th style="padding-right:8px;">dist (m)</th><th>light</th><th></th>
        </tr>
        {body}
      </table>
    </div>
    """


# ── single-map render (used by --start/--end + --layout layered) ───────────
def _bounds_for_routes(short: RouteResult, lit: RouteResult, pad: float = 0.0005):
    pts = short.coords + lit.coords
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return [[min(lats) - pad, min(lons) - pad], [max(lats) + pad, max(lons) + pad]]


def _render_single(G, edge_scores, edge_sources, pair, lambda_, unknown_penalty,
                   mode, metric, fit_to_pair: bool = True):
    """Build one folium map for a single pair. Returns (Map, short, lit)."""
    import folium

    title, start, end = pair
    short, lit = compute_routes(G, edge_scores, start, end, lambda_=lambda_,
                                unknown_penalty=unknown_penalty,
                                edge_sources=edge_sources)

    # centre on the midpoint of the pair
    centre = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    fmap = folium.Map(location=centre, zoom_start=16, tiles=DARK_TILE,
                      control_scale=True)
    _add_basemap(fmap, G, edge_scores, edge_sources=edge_sources)
    _add_route(fmap, short, ROUTE_SHORTEST_COLOR)
    _add_route(fmap, lit, ROUTE_LIT_COLOR, dash_array=ROUTE_LIT_DASH)
    _add_markers(fmap, start, end, title)

    if fit_to_pair:
        fmap.fit_bounds(_bounds_for_routes(short, lit))

    return fmap, short, lit


# ── grid layout (the report figure) ────────────────────────────────────────
def render_grid(G, edge_scores, edge_sources, pairs, lambda_, unknown_penalty,
                mode, metric, out: Path):
    """Three side-by-side mini-maps, one per pair. Each map auto-zooms to its
    pair. Title strip at the top, per-panel caption under each map.
    """
    panel_htmls = []
    for i, pair in enumerate(pairs):
        accent = PAIR_ACCENTS[i % len(PAIR_ACCENTS)]
        fmap, short, lit = _render_single(G, edge_scores, edge_sources, pair,
                                          lambda_, unknown_penalty, mode, metric,
                                          fit_to_pair=True)
        for line in [f"[{pair[0]}] λ={lambda_}"] + summary_lines(short, lit):
            print(line)
        # take only the body of each folium map (not <html><head>...)
        map_html = fmap._repr_html_()
        cap = _panel_caption_html(pair[0], short, lit, accent)
        panel_htmls.append(
            f'<div class="panel" style="flex:1 1 0; min-width:280px;">'
            f'<div class="map-wrap" style="height:520px;">{map_html}</div>'
            f'{cap}'
            f'</div>'
        )

    title_strip = _title_strip_html(mode, metric, lambda_, unknown_penalty, len(pairs))
    page = f"""<!doctype html>
<html><head>
  <meta charset="utf-8" />
  <title>NightWalk · routes (grid)</title>
  <style>
    html, body {{ margin:0; padding:0; background:#0c0f14; color:#e6e9ef;
                  font:14px/1.4 system-ui, -apple-system, sans-serif; }}
    .grid {{ display:flex; flex-wrap:wrap; gap:16px; padding:16px;
            align-items:stretch; }}
    .panel {{ background:#11151c; border:1px solid #1f2530; border-radius:8px;
              overflow:hidden; box-shadow:0 1px 6px rgba(0,0,0,0.4);
              display:flex; flex-direction:column; }}
    .map-wrap iframe {{ width:100%; height:100%; border:0; }}
    /* folium injects its own height — make sure the iframe fills the wrapper */
    .map-wrap > div, .map-wrap iframe {{ height:100% !important; width:100% !important; }}
  </style>
</head>
<body>
  {title_strip}
  <div class="grid">
    {''.join(panel_htmls)}
  </div>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[render_map] wrote {out}")


# ── layered layout (interactive — one map, toggleable pairs) ───────────────
def render_layered(G, edge_scores, edge_sources, pairs, lambda_, unknown_penalty,
                   mode, metric, out: Path):
    """One folium map; each pair lives in its own FeatureGroup behind a
    LayerControl. Default: pair 1 visible, others hidden."""
    import folium

    all_pts = [pt for _, a, b in pairs for pt in (a, b)]
    centre = (sum(p[0] for p in all_pts) / len(all_pts),
              sum(p[1] for p in all_pts) / len(all_pts))

    fmap = folium.Map(location=centre, zoom_start=15, tiles=DARK_TILE,
                      control_scale=True)
    _add_basemap(fmap, G, edge_scores, edge_sources=edge_sources)

    legend_rows = []
    for i, (title, start, end) in enumerate(pairs):
        accent = PAIR_ACCENTS[i % len(PAIR_ACCENTS)]
        short, lit = compute_routes(G, edge_scores, start, end, lambda_=lambda_,
                                    unknown_penalty=unknown_penalty,
                                    edge_sources=edge_sources)
        for line in [f"[{title}] λ={lambda_}"] + summary_lines(short, lit):
            print(line)
        group = folium.FeatureGroup(name=title, show=(i == 0))
        _add_route(group, short, ROUTE_SHORTEST_COLOR)
        _add_route(group, lit, ROUTE_LIT_COLOR, dash_array=ROUTE_LIT_DASH)
        _add_markers(group, start, end, title, accent=accent)
        group.add_to(fmap)
        legend_rows.append((title, short, lit))

    folium.LayerControl(collapsed=False).add_to(fmap)

    # Top title strip + overall table at the bottom-left
    root = fmap.get_root()
    root.html.add_child(folium.Element(
        _title_strip_html(mode, metric, lambda_, unknown_penalty, len(pairs))))
    root.html.add_child(folium.Element(_overall_table_html(legend_rows)))

    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"[render_map] wrote {out}")


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_latlon(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output path; used for --layout=layered and single-pair mode. "
                         "In --layout=both, also produces *_grid.html alongside.")
    ap.add_argument("--start", type=str, default=None, help="lat,lon")
    ap.add_argument("--end", type=str, default=None, help="lat,lon")
    ap.add_argument("--examples", action="store_true",
                    help="render the canned EXAMPLE_PAIRS gallery for §5.2")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=2.0)
    ap.add_argument("--mu-unk", dest="unknown_penalty", type=float, default=0.5,
                    help="extra multiplicative cost for fallback (unknown) units (default 0.5)")
    ap.add_argument("--mode", choices=["edge", "block"], default="edge",
                    help="which aggregation to read (default: edge)")
    ap.add_argument("--metric", choices=["median", "dark"], default="median",
                    help="which lighting statistic to use (default: median)")
    ap.add_argument("--layout", choices=["grid", "layered", "both"], default="both",
                    help="for --examples: 'grid' = report figure, 'layered' = live demo, 'both' = emit both")
    args = ap.parse_args()

    if not args.graph.exists() or not args.scores.exists():
        raise SystemExit("missing graph or edge_lighting.csv — run build_graph.py and aggregate_to_edges.py first")

    import osmnx as ox
    G = ox.load_graphml(args.graph)
    for _, data in G.nodes(data=True):
        data["x"] = float(data["x"])
        data["y"] = float(data["y"])
    for _, _, _, data in G.edges(keys=True, data=True):
        if "length" in data:
            data["length"] = float(data["length"])

    edge_scores = load_edge_scores(args.scores, mode=args.mode, metric=args.metric)
    edge_sources = load_edge_sources(args.scores, mode=args.mode)

    print(f"[render_map] mode={args.mode}  metric={args.metric}  "
          f"λ={args.lambda_}  μ_unk={args.unknown_penalty}")

    if args.examples:
        pairs = EXAMPLE_PAIRS
        if args.layout in ("grid", "both"):
            grid_out = (args.out.with_name(args.out.stem + "_grid.html")
                        if args.layout == "both" else args.out)
            # if user passed --out explicitly with layout=both, place grid sibling next to it
            if args.layout == "both":
                grid_out = DEFAULT_GRID_OUT if args.out == DEFAULT_OUT else grid_out
            render_grid(G, edge_scores, edge_sources, pairs,
                        args.lambda_, args.unknown_penalty,
                        args.mode, args.metric, grid_out)
        if args.layout in ("layered", "both"):
            layered_out = (args.out.with_name(args.out.stem + "_layered.html")
                           if args.layout == "both" else args.out)
            if args.layout == "both":
                layered_out = DEFAULT_LAYERED_OUT if args.out == DEFAULT_OUT else layered_out
            render_layered(G, edge_scores, edge_sources, pairs,
                           args.lambda_, args.unknown_penalty,
                           args.mode, args.metric, layered_out)
        return

    if not (args.start and args.end):
        raise SystemExit("must pass --start and --end (or --examples)")
    pair = ("custom", _parse_latlon(args.start), _parse_latlon(args.end))
    render_layered(G, edge_scores, edge_sources, [pair],
                   args.lambda_, args.unknown_penalty,
                   args.mode, args.metric, args.out)


if __name__ == "__main__":
    main()
