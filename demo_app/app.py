"""app.py — Streamlit demo for NightWalk lighting-aware routing.

Run from repo root:

    streamlit run demo_app/app.py

Three independent toggles in the sidebar control behaviour:

  • Aggregation: **per edge / per block** — which set of scores drives routing
    + basemap + audit.
  • Lighting metric: **Typical (median) / Pessimistic (worst quartile)** —
    typical optimises for average brightness; pessimistic for the dimmest
    stretch you'll walk through.
  • Click mode: **Set endpoints / Inspect edge** — what map clicks do.

λ tunes the lighting weight in the route cost; μ_unk extra-penalises units
that lack direct evidence ("avoid unknown areas").
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route import (  # noqa: E402
    compute_routes,
    load_edge_predictions,
    load_edge_table,
    summary_lines,
)
from render_map import (  # noqa: E402
    DARK_TILE,
    FALLBACK_GRAY,
    ROUTE_LIT_COLOR,
    ROUTE_LIT_DASH,
    ROUTE_SHORTEST_COLOR,
    _add_basemap,
    _add_markers,
    _add_route,
    _color_for,
)


def _walk_up_for(*targets) -> list[Path]:
    """Return any parent directory that contains one of the target sub-paths."""
    found = []
    cur = Path(__file__).resolve().parent
    for _ in range(8):
        for target in targets:
            cand = cur / target
            if cand.is_dir():
                found.append(cand)
        if cur.parent == cur:
            break
        cur = cur.parent
    seen, uniq = set(), []
    for p in found:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_GRAPH = DEMO_DATA / "wsp_walk.graphml"
DEFAULT_SCORES = DEMO_DATA / "edge_lighting.csv"
DEFAULT_EDGE_PREDS = DEMO_DATA / "edge_predictions.csv"

IMAGE_DIR_CANDIDATES = _walk_up_for(
    "urban-mosaic/washington-square",
    "nightwalk-images-224",
)

st.set_page_config(page_title="NightWalk · safe routing demo", layout="wide")


# ── data loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading graph + lighting scores…")
def _load(graph_path: str, scores_path: str, edge_preds_path: str):
    import osmnx as ox

    G = ox.load_graphml(graph_path)
    for _, data in G.nodes(data=True):
        data["x"] = float(data["x"])
        data["y"] = float(data["y"])
    for _, _, _, data in G.edges(keys=True, data=True):
        if "length" in data:
            data["length"] = float(data["length"])

    edge_table = load_edge_table(Path(scores_path))
    edge_preds = load_edge_predictions(Path(edge_preds_path))
    return G, edge_table, edge_preds


def _scores_for(edge_table, mode: str, metric: str):
    """Derive {(u,v,k) → score} and {(u,v,k) → source} for active mode × metric."""
    if metric == "median":
        score_col = f"lighting_score_{mode}"
    else:  # "dark"
        score_col = f"lighting_dark_score_{mode}"
    src_col = f"source_{mode}"
    scores = {k: row[score_col] for k, row in edge_table.items()}
    sources = {k: row[src_col] for k, row in edge_table.items()}
    return scores, sources


# ── click helpers ──────────────────────────────────────────────────────────
def _snap_to_edge(G, lat: float, lon: float):
    import osmnx as ox
    u, v, k = ox.distance.nearest_edges(G, X=lon, Y=lat)
    return (int(u), int(v), int(k))


# ── thumbnails ─────────────────────────────────────────────────────────────
def _resolve_image_path(image_id: str) -> Path | None:
    for base in IMAGE_DIR_CANDIDATES:
        p = base / image_id
        if p.exists():
            return p
    return None


def _representative_contributor(edge_preds, edge_table, edge_key):
    """Pick the predictor whose raw_pred is closest to the edge's median."""
    rows = edge_preds.get(edge_key, [])
    if not rows:
        return None
    target = edge_table.get(edge_key, {}).get("lighting_raw_edge", 0.0)
    return min(rows, key=lambda r: abs(r["raw_pred"] - target))


def _route_edges_in_order(G, route):
    out = []
    nodes = route.nodes
    for u, v in zip(nodes[:-1], nodes[1:]):
        d = G.get_edge_data(u, v) or {}
        if not d:
            continue
        k = min(d, key=lambda kk: d[kk].get("length", float("inf")))
        out.append((int(u), int(v), int(k)))
    return out


def _placeholder_caption(rep, source: str) -> str:
    if rep is None:
        if source == "fallback":
            return "no data — dataset median used"
        return "inferred — no direct evidence"
    short_id = rep["image_id"].rsplit("/", 1)[-1]
    return f"image not found locally\n`{short_id}`"


def _render_route_card(col, idx, edge_key, edge_table, edge_preds, G,
                       agg_mode: str, metric: str):
    row = edge_table.get(edge_key, {})
    source = row.get(f"source_{agg_mode}", "?")
    score_col = (f"lighting_score_{agg_mode}" if metric == "median"
                 else f"lighting_dark_score_{agg_mode}")
    score = row.get(score_col, 0.0)
    length_m = float((G.get_edge_data(*edge_key) or {}).get("length", 0.0))
    rep = _representative_contributor(edge_preds, edge_table, edge_key)

    swatch = _color_for(score) if source != "fallback" else FALLBACK_GRAY
    metric_label = "med" if metric == "median" else "p25"
    col.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<b>{idx + 1}.</b>"
        f"<span style='display:inline-block;width:14px;height:10px;background:{swatch};"
        f"border:1px solid #888;'></span>"
        f"<span>{agg_mode} {metric_label} <b>{score:.2f}</b>  ·  {length_m:.0f} m  ·  "
        f"<span style='color:#666;font-size:11px;'>{source}</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if rep is not None:
        path = _resolve_image_path(rep["image_id"])
        if path is not None:
            col.image(str(path), use_container_width=True)
        else:
            col.caption(_placeholder_caption(rep, source))
            col.caption(f"raw_pred={rep['raw_pred']:.2f}  d={rep['edge_distance_m']:.1f} m")
    else:
        col.caption(_placeholder_caption(None, source))


def _render_route_thumbnails(G, short_route, lit_route, edge_table, edge_preds,
                             agg_mode: str, metric: str, max_per_route: int = 12):
    short_edges = _route_edges_in_order(G, short_route)
    lit_edges = _route_edges_in_order(G, lit_route)

    st.markdown("### Per-segment comparison")
    st.caption(
        "Each card shows one edge along the route in walking order. "
        "The thumbnail is the daytime image whose prediction was closest to "
        "the per-edge median. If the source image isn't on disk locally a "
        "placeholder appears — drop the dataset under "
        "`urban-mosaic/washington-square/` or `nightwalk-images-224/` and the "
        "thumbnails appear automatically."
    )

    full_short = len(short_edges)
    full_lit = len(lit_edges)
    if max_per_route and (full_short > max_per_route or full_lit > max_per_route):
        st.caption(f"Showing the first **{max_per_route}** segments of each route.")
        short_edges = short_edges[:max_per_route]
        lit_edges = lit_edges[:max_per_route]

    short_col, lit_col = st.columns(2)
    short_col.markdown(
        f"<div style='border-left:4px solid {ROUTE_SHORTEST_COLOR};padding-left:8px;'>"
        f"<b>Shortest</b> · {short_route.distance_m:.0f} m · light {short_route.mean_lighting:.2f} "
        f"({len(short_edges)} / {full_short} segments shown)"
        f"</div>", unsafe_allow_html=True)
    detour = (lit_route.distance_m / short_route.distance_m) if short_route.distance_m else 1.0
    lit_col.markdown(
        f"<div style='border-left:4px solid {ROUTE_LIT_COLOR};padding-left:8px;'>"
        f"<b>Well-lit</b> · {lit_route.distance_m:.0f} m ({detour:.2f}×) · "
        f"light {lit_route.mean_lighting:.2f} "
        f"({len(lit_edges)} / {full_lit} segments shown)"
        f"</div>", unsafe_allow_html=True)

    for i in range(max(len(short_edges), len(lit_edges))):
        if i < len(short_edges):
            _render_route_card(short_col, i, short_edges[i], edge_table, edge_preds,
                               G, agg_mode, metric)
        else:
            short_col.markdown("&nbsp;", unsafe_allow_html=True)
        if i < len(lit_edges):
            _render_route_card(lit_col, i, lit_edges[i], edge_table, edge_preds,
                               G, agg_mode, metric)
        else:
            lit_col.markdown("&nbsp;", unsafe_allow_html=True)


# ── inspector panel (now also shows p25 alongside median) ──────────────────
SRC_EXPLAIN = {
    "direct":              "median of the rows above (or pooled across the block's edges)",
    "smoothed_same_name":  "no direct evidence — averaged over adjacent *same-name* units",
    "smoothed_other":      "no direct evidence and no same-name neighbour — averaged over any adjacent unit",
    "fallback":            "no direct or neighbour evidence — using dataset median",
}


def _render_inspector_panel(edge_key, edge_table, edge_preds, G):
    row = edge_table.get(edge_key)
    if row is None:
        st.warning(f"edge {edge_key} not found in edge_lighting.csv")
        return

    u, v, k = edge_key
    block_id = row["block_id"]
    st.markdown(f"### Edge `{u}` → `{v}` (key {k}) · block `#{block_id}`")

    contributors = edge_preds.get(edge_key, [])
    st.markdown(f"**1 — predictions within R m of this OSM edge** ({len(contributors)})")
    if contributors:
        contributors_sorted = sorted(contributors, key=lambda r: r["raw_pred"])
        st.dataframe(
            [{"image_id": c["image_id"][:60] + ("…" if len(c["image_id"]) > 60 else ""),
              "raw_pred": round(c["raw_pred"], 3),
              "dist_m": round(c["edge_distance_m"], 2)} for c in contributors_sorted],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No predictions within R m of this OSM edge.")

    st.markdown("**2 — aggregations: median & p25 × edge & block**")

    # Two columns: edge vs block; each shows BOTH median and p25
    cE, cB = st.columns(2)

    def swatch(score, source):
        bg = _color_for(score) if source != "fallback" else FALLBACK_GRAY
        return (f"<span style='display:inline-block;width:14px;height:10px;background:{bg};"
                f"border:1px solid #888;vertical-align:middle;margin-right:4px;'></span>")

    cE.markdown(
        f"**Per OSM edge** · contributors **{row['n_predictions_edge']}** · "
        f"source **{row['source_edge']}**\n\n"
        f"<span style='color:#7e8a9c;font-size:11px;'>{SRC_EXPLAIN.get(row['source_edge'], '')}</span>\n\n"
        f"{swatch(row['lighting_score_edge'], row['source_edge'])} "
        f"median: raw `{row['lighting_raw_edge']:.3f}` · score **`{row['lighting_score_edge']:.3f}`**  \n"
        f"{swatch(row['lighting_dark_score_edge'], row['source_edge'])} "
        f"p25 (dark): raw `{row['lighting_dark_raw_edge']:.3f}` · score **`{row['lighting_dark_score_edge']:.3f}`**",
        unsafe_allow_html=True,
    )
    cB.markdown(
        f"**Per block (#{block_id})** · contributors **{row['n_predictions_block']}** · "
        f"source **{row['source_block']}**\n\n"
        f"<span style='color:#7e8a9c;font-size:11px;'>{SRC_EXPLAIN.get(row['source_block'], '')}</span>\n\n"
        f"{swatch(row['lighting_score_block'], row['source_block'])} "
        f"median: raw `{row['lighting_raw_block']:.3f}` · score **`{row['lighting_score_block']:.3f}`**  \n"
        f"{swatch(row['lighting_dark_score_block'], row['source_block'])} "
        f"p25 (dark): raw `{row['lighting_dark_raw_block']:.3f}` · score **`{row['lighting_dark_score_block']:.3f}`**",
        unsafe_allow_html=True,
    )

    st.markdown("**3 — how routing uses this edge**")
    try:
        length_m = float(G.get_edge_data(u, v, k).get("length", 0.0))
    except Exception:
        length_m = 0.0
    st.markdown(
        f"`lit_cost = length · (1 + λ·(1 − score)) · (1 + μ_unk·is_fallback)` &nbsp; "
        f"using the active *aggregation × metric* you picked in the sidebar. "
        f"length here = **{length_m:.1f} m**. Darker units force detours; "
        f"fallback units do too when μ_unk > 0."
    )


def _render_audit_expander(edge_table, agg_mode: str, metric: str):
    import pandas as pd

    label = f"{agg_mode}-level · {metric}"
    with st.expander(f"Pipeline audit — {label} stats", expanded=False):
        df = pd.DataFrame(edge_table.values())
        if df.empty:
            st.info("No edge_lighting.csv data to summarise.")
            return
        src_col = f"source_{agg_mode}"
        score_col = (f"lighting_score_{agg_mode}" if metric == "median"
                     else f"lighting_dark_score_{agg_mode}")
        n_col = f"n_predictions_{agg_mode}"

        if agg_mode == "block":
            df = df.drop_duplicates(subset="block_id")

        cov = df[src_col].value_counts().to_dict()
        total = len(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{agg_mode.capitalize()}s (total)", f"{total}")
        c2.metric("Direct", f"{cov.get('direct', 0)}",
                  delta=f"{100*cov.get('direct', 0)/max(1,total):.0f} %", delta_color="off")
        c3.metric("Smoothed same-name", f"{cov.get('smoothed_same_name', 0)}",
                  delta=f"{100*cov.get('smoothed_same_name', 0)/max(1,total):.0f} %", delta_color="off")
        c4.metric("Smoothed other", f"{cov.get('smoothed_other', 0)}",
                  delta=f"{100*cov.get('smoothed_other', 0)/max(1,total):.0f} %", delta_color="off")
        c5.metric("Fallback", f"{cov.get('fallback', 0)}",
                  delta=f"{100*cov.get('fallback', 0)/max(1,total):.0f} %", delta_color="off")

        st.markdown(f"**Lighting score distribution ({metric})**")
        st.bar_chart(df[score_col].value_counts(bins=20, sort=False))

        st.markdown(f"**Direct contributors per {agg_mode}** (units with ≥1 direct contributor)")
        direct = df[df[src_col] == "direct"]
        if not direct.empty:
            st.bar_chart(direct[n_col].value_counts().sort_index())
        else:
            st.info("No direct-evidence units.")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    import folium
    from streamlit_folium import st_folium

    st.title("NightWalk · lighting-aware route planning")
    st.caption(
        "Two routes between the same endpoints. **Blue solid** = shortest path "
        "(distance-only). **Hot-pink dashed** = well-lit path (penalises dark "
        "units using NightWalk's predicted street brightness)."
    )

    if not DEFAULT_GRAPH.exists() or not DEFAULT_SCORES.exists():
        st.error(
            "Missing demo data. From the repo root run:\n\n"
            "```\npython demo_app/build_graph.py\n"
            "python demo_app/build_predictions.py\n"
            "python demo_app/aggregate_to_edges.py\n```"
        )
        return

    G, edge_table, edge_preds = _load(
        str(DEFAULT_GRAPH), str(DEFAULT_SCORES), str(DEFAULT_EDGE_PREDS))

    # session state defaults
    for k, v in (("click_mode", "Set endpoints"), ("agg_mode", "edge"),
                 ("metric", "median"),
                 ("start", None), ("end", None),
                 ("inspected_edge", None), ("last_click", None)):
        st.session_state.setdefault(k, v)

    # ── sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Controls")

        agg_label = st.radio(
            "Aggregation",
            ["per edge", "per block"],
            horizontal=True, key="agg_mode_radio",
            index=0 if st.session_state.agg_mode == "edge" else 1,
            help="Per-edge: one score per OSM walking-graph edge. Per-block: "
                 "pool predictions across an entire named-street segment.",
        )
        st.session_state.agg_mode = "edge" if agg_label == "per edge" else "block"

        metric_label = st.radio(
            "Lighting metric",
            ["Typical (median)", "Pessimistic (p25)"],
            horizontal=False, key="metric_radio",
            index=0 if st.session_state.metric == "median" else 1,
            help="Typical: optimise for average brightness along the route. "
                 "Pessimistic: optimise so the darkest stretch you walk through "
                 "is as bright as possible.",
        )
        st.session_state.metric = "median" if metric_label.startswith("Typical") else "dark"

        st.session_state.click_mode = st.radio(
            "Click mode", ["Set endpoints", "Inspect edge"],
            horizontal=False, key="click_mode_radio",
            index=["Set endpoints", "Inspect edge"].index(st.session_state.click_mode),
        )

        st.divider()
        lambda_ = st.slider("λ — lighting weight", 0.0, 8.0, 2.0, 0.5,
                            help="0 = ignore lighting (shortest path only). "
                                 "Higher = bigger detours to stay on well-lit streets.")
        mu_unk = st.slider("μ_unk — unknown-area penalty", 0.0, 2.0, 0.5, 0.1,
                           help="Extra multiplicative cost applied to 'fallback' units "
                                "(no direct or neighbour evidence). 0 disables.")

        st.divider()
        st.markdown(
            f"**Start**: `{st.session_state.start}`\n\n"
            f"**End**: `{st.session_state.end}`"
        )
        if st.button("Reset endpoints", use_container_width=True):
            st.session_state.start = None
            st.session_state.end = None
            st.session_state.inspected_edge = None
            st.rerun()

    # subtitle bar describing current settings
    st.markdown(
        f"<div style='color:#7e8a9c;font-size:13px;margin:-6px 0 12px 0;'>"
        f"routing on <b>{st.session_state.agg_mode}</b> × "
        f"<b>{'typical (median)' if st.session_state.metric == 'median' else 'pessimistic (p25)'}</b> "
        f"with λ = <b>{lambda_}</b>, μ_unk = <b>{mu_unk}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # derive scores for the active aggregation × metric
    edge_scores, edge_sources = _scores_for(edge_table, st.session_state.agg_mode,
                                            st.session_state.metric)

    # ── build the map ───────────────────────────────────────────────────────
    centre = (40.7308, -73.9960)
    fmap = folium.Map(location=centre, zoom_start=16, tiles=DARK_TILE,
                      control_scale=True)
    _add_basemap(fmap, G, edge_scores, edge_sources=edge_sources)

    routes = None
    if st.session_state.start and st.session_state.end:
        short, lit = compute_routes(G, edge_scores, st.session_state.start,
                                    st.session_state.end, lambda_=lambda_,
                                    unknown_penalty=mu_unk, edge_sources=edge_sources)
        _add_route(fmap, short, ROUTE_SHORTEST_COLOR)
        _add_route(fmap, lit, ROUTE_LIT_COLOR, dash_array=ROUTE_LIT_DASH)
        _add_markers(fmap, st.session_state.start, st.session_state.end, "selected")
        routes = (short, lit)
    elif st.session_state.start:
        folium.Marker(st.session_state.start, tooltip="start — click again for end",
                      icon=folium.Icon(color="blue")).add_to(fmap)

    if st.session_state.inspected_edge is not None:
        u, v, k = st.session_state.inspected_edge
        data = G.get_edge_data(u, v, k) or {}
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G.nodes[u]["y"], G.nodes[u]["x"]),
                      (G.nodes[v]["y"], G.nodes[v]["x"])]
        folium.PolyLine(coords, color="#ffffff", weight=6, opacity=0.85).add_to(fmap)

    # main two-column layout: comparison/inspector ⟂ map
    col_left, col_right = st.columns([1, 3])
    with col_right:
        map_state = st_folium(fmap, height=620, width=None,
                              returned_objects=["last_clicked"])

    clicked = (map_state or {}).get("last_clicked")
    if clicked:
        token = (round(clicked["lat"], 7), round(clicked["lng"], 7))
        if token != st.session_state.last_click:
            st.session_state.last_click = token
            if st.session_state.click_mode == "Set endpoints":
                point = (clicked["lat"], clicked["lng"])
                if st.session_state.start is None:
                    st.session_state.start = point
                    st.rerun()
                elif st.session_state.end is None:
                    st.session_state.end = point
                    st.rerun()
            else:  # Inspect edge
                st.session_state.inspected_edge = _snap_to_edge(
                    G, clicked["lat"], clicked["lng"])
                st.rerun()

    with col_left:
        if routes is not None:
            short, lit = routes
            st.markdown("### Route comparison")
            st.metric("Shortest route", f"{short.distance_m:.0f} m",
                      delta=f"mean lighting {short.mean_lighting:.2f}", delta_color="off")
            st.metric("Well-lit route", f"{lit.distance_m:.0f} m",
                      delta=f"mean lighting {lit.mean_lighting:.2f}", delta_color="off")
            detour = (lit.distance_m / short.distance_m) if short.distance_m else 1.0
            st.metric("Detour", f"{detour:.2f}×",
                      delta=f"Δ light {lit.mean_lighting - short.mean_lighting:+.2f}",
                      delta_color="off")
            st.code("\n".join(summary_lines(short, lit)))
        elif st.session_state.click_mode == "Set endpoints":
            st.info("Click the map to set the start, then click again to set the end.")
        else:
            st.info("Switch click mode to **Set endpoints** to plan a route, "
                    "or click any street with **Inspect edge** on.")

        if st.session_state.click_mode == "Inspect edge":
            if st.session_state.inspected_edge is not None:
                st.divider()
                _render_inspector_panel(
                    st.session_state.inspected_edge, edge_table, edge_preds, G)
            else:
                st.info("Click any street on the map to see its lighting breakdown.")

        st.divider()
        _render_audit_expander(edge_table, st.session_state.agg_mode, st.session_state.metric)

    if routes is not None:
        st.divider()
        _render_route_thumbnails(G, routes[0], routes[1], edge_table, edge_preds,
                                 st.session_state.agg_mode, st.session_state.metric)


if __name__ == "__main__":
    main()
