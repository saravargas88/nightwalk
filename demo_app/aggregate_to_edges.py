"""aggregate_to_edges.py — predictions.csv → per-edge AND per-block lighting scores.

Implements §5.1 of the NightWalk report. Always emits two aggregations in the
same output CSV so the demo can compare them via the UI toggle:

  • per-edge   — pool predictions within --radius-m of each OSM edge's geometry
  • per-block  — pool predictions across all edges that form one logical
                 "block" (named-street run between real intersections)

Pipeline
--------
1. **Spatial-radius aggregation** (v3): for each OSM edge, find every prediction
   within R metres of any point along the edge's geometry (KDTree query).
   This replaces the v2 "nearest-edge snap within tolerance" — fixes the
   coverage artifact where many tiny edges had zero contributors because each
   prediction snapped to only ONE of the ~4 short edges near it.
2. Identify blocks: group named edges, split at degree-≥3 intersections.
   Unnamed edges (crosswalks, plaza paths) attach to their longest adjacent
   named block.
3. Aggregate per edge AND per block: compute both `median` and `p25` (the
   25th-percentile — captures the *darkest stretch* a pedestrian walks).
4. For the rare unit still without direct evidence:
     a) same-name neighbour smoothing (prefer same OSM name);
     b) any-neighbour smoothing;
     c) dataset-median fallback.
5. Normalise each population (median × edge, median × block, p25 × edge,
   p25 × block) independently via percentile (default) or robust-sigmoid.

Output: data/edge_lighting.csv — one row per OSM edge with `block_id` and four
parallel score column sets. Plus data/edge_predictions.csv (image → edge
sidecar) for the inspector/thumbnail UIs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEMO_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_PREDS = DEMO_DATA / "predictions.csv"
DEFAULT_GRAPH = DEMO_DATA / "wsp_walk.graphml"
DEFAULT_OUT = DEMO_DATA / "edge_lighting.csv"
DEFAULT_EDGE_PREDS_OUT = DEMO_DATA / "edge_predictions.csv"


# ── prediction loader ──────────────────────────────────────────────────────
def _read_predictions(path: Path):
    ids, xs, ys, vals = [], [], [], []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                xs.append(float(r["lon"]))
                ys.append(float(r["lat"]))
                vals.append(float(r["predicted_brightness"]))
            except (KeyError, ValueError):
                continue
            ids.append(r.get("image_id", ""))
    return ids, xs, ys, vals


# ── normalisation ──────────────────────────────────────────────────────────
def _percentile_normalize(values, p_lo: float = 5.0, p_hi: float = 95.0):
    """Linear map of p_lo→0, p_hi→1; clipped at the tails."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    span = max(hi - lo, 1e-9)
    scores = np.clip((arr - lo) / span, 0.0, 1.0)
    return scores, float(lo), float(hi)


def _robust_sigmoid_normalize(values):
    """Median/MAD z-score → logistic sigmoid into (0, 1). Legacy default."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad if mad > 1e-9 else (float(np.std(arr)) or 1.0)
    z = (arr - med) / scale
    return 1.0 / (1.0 + np.exp(-z)), med, scale


def _normalize(values, mode: str):
    if mode == "percentile":
        scores, lo, hi = _percentile_normalize(values)
        return scores, f"p5={lo:.3f} p95={hi:.3f}"
    elif mode == "sigmoid":
        scores, med, scale = _robust_sigmoid_normalize(values)
        return scores, f"median={med:.3f} robust_scale={scale:.3f}"
    else:
        raise ValueError(f"unknown --norm {mode!r}")


# ── block identification (unchanged from v2) ───────────────────────────────
def _normalize_name(name):
    if isinstance(name, list):
        return name[0] if name else None
    return name or None


def identify_blocks(G):
    """Assign a block_id to every edge in the multigraph. See v2 for details."""
    Gu = G.to_undirected()
    node_deg = dict(Gu.degree())

    all_edges = list(G.edges(keys=True))
    incident: dict[int, list[tuple]] = defaultdict(list)
    for u, v, k in all_edges:
        incident[u].append((u, v, k))
        incident[v].append((u, v, k))

    edges_by_name: dict[str | None, list[tuple]] = defaultdict(list)
    for u, v, k, data in G.edges(keys=True, data=True):
        name = _normalize_name(data.get("name"))
        edges_by_name[name].append((u, v, k))

    block_of_edge: dict[tuple, int] = {}
    blocks: dict[int, list[tuple]] = {}
    next_id = 0

    for name, name_edges in edges_by_name.items():
        if name is None:
            continue
        adj: dict[int, list[tuple]] = defaultdict(list)
        for u, v, k in name_edges:
            adj[u].append((u, v, k))
            adj[v].append((u, v, k))

        unvisited = set(name_edges)
        while unvisited:
            seed = next(iter(unvisited))
            unvisited.discard(seed)
            block_edges = [seed]
            stack = [seed]
            while stack:
                u, v, k = stack.pop()
                for end in (u, v):
                    if node_deg.get(end, 0) >= 3:
                        continue
                    for cand in adj[end]:
                        if cand in unvisited:
                            unvisited.discard(cand)
                            block_edges.append(cand)
                            stack.append(cand)
            for e in block_edges:
                block_of_edge[e] = next_id
            blocks[next_id] = block_edges
            next_id += 1

    def _length(e):
        u, v, k = e
        return float(G.get_edge_data(u, v, k).get("length", 0.0))

    for edge in edges_by_name.get(None, []):
        u, v, k = edge
        candidates = []
        for end in (u, v):
            for nb in incident[end]:
                if nb == edge or nb not in block_of_edge:
                    continue
                candidates.append((_length(nb), nb))
        if candidates:
            candidates.sort(reverse=True)
            _, best_named = candidates[0]
            bid = block_of_edge[best_named]
            block_of_edge[edge] = bid
            blocks[bid].append(edge)
        else:
            block_of_edge[edge] = next_id
            blocks[next_id] = [edge]
            next_id += 1

    return block_of_edge, blocks


# ── neighbour smoothing + fallback (v3: same-name preference) ─────────────
def _smooth_and_fill(direct_raw: dict, all_units, neighbours_of: dict,
                     names_of: dict):
    """Fill empty units with same-name → any-neighbour → dataset-median.

    Returns (raw_dict, source_dict, counts) where source ∈ {direct,
    smoothed_same_name, smoothed_other, fallback}.
    """
    import numpy as np

    raw = dict(direct_raw)
    sources = {u: "direct" for u in direct_raw}
    counts = {"smoothed_same_name": 0, "smoothed_other": 0, "fallback": 0}

    for unit in all_units:
        if unit in raw:
            continue
        my_name = names_of.get(unit)
        # try same-name neighbours first
        same_name_vals = [
            direct_raw[nb] for nb in neighbours_of.get(unit, ())
            if nb in direct_raw and names_of.get(nb) is not None
            and names_of.get(nb) == my_name
        ]
        if same_name_vals:
            raw[unit] = float(np.mean(same_name_vals))
            sources[unit] = "smoothed_same_name"
            counts["smoothed_same_name"] += 1
            continue
        # fall back to all neighbours
        nb_vals = [direct_raw[nb] for nb in neighbours_of.get(unit, ())
                   if nb in direct_raw]
        if nb_vals:
            raw[unit] = float(np.mean(nb_vals))
            sources[unit] = "smoothed_other"
            counts["smoothed_other"] += 1
            continue
        # nothing → dataset-median fallback
        raw[unit] = (float(np.median(list(direct_raw.values())))
                     if direct_raw else 0.0)
        sources[unit] = "fallback"
        counts["fallback"] += 1

    return raw, sources, counts


# ── spatial-radius aggregation ──────────────────────────────────────────────
def _edge_sample_points(Gp, edge, step_m: float = 5.0):
    """Sample points along an edge's geometry every `step_m` metres.

    Returns a list of (x, y) tuples in the projected CRS.
    """
    u, v, k = edge
    data = Gp.get_edge_data(u, v, k) or {}
    if "geometry" in data:
        line = data["geometry"]
        try:
            length = float(line.length)
        except Exception:
            length = 0.0
        n = max(2, int(length / step_m) + 1)
        out = []
        for i in range(n):
            t = i / (n - 1)
            try:
                p = line.interpolate(t, normalized=True)
                out.append((float(p.x), float(p.y)))
            except Exception:
                continue
        return out, line
    # fall back to endpoints + midpoint
    try:
        ux, uy = float(Gp.nodes[u]["x"]), float(Gp.nodes[u]["y"])
        vx, vy = float(Gp.nodes[v]["x"]), float(Gp.nodes[v]["y"])
    except KeyError:
        return [], None
    midx, midy = (ux + vx) / 2, (uy + vy) / 2
    return [(ux, uy), (midx, midy), (vx, vy)], None


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--edge-preds-out", type=Path, default=DEFAULT_EDGE_PREDS_OUT)
    ap.add_argument("--radius-m", type=float, default=30.0,
                    help="pool predictions within this many metres of each edge's geometry")
    ap.add_argument("--norm", choices=["percentile", "sigmoid"], default="percentile",
                    help="how to map raw aggregates into [0, 1] (default: percentile)")
    args = ap.parse_args()

    import numpy as np
    import osmnx as ox
    from scipy.spatial import cKDTree
    import geopandas as gpd
    from shapely.geometry import Point

    if not args.graph.exists():
        raise SystemExit(f"graph not found: {args.graph} — run build_graph.py first")
    if not args.predictions.exists():
        raise SystemExit(f"predictions not found: {args.predictions} — run build_predictions.py first")

    print(f"[aggregate] loading graph: {args.graph}")
    G = ox.load_graphml(args.graph)
    ids, xs, ys, vals = _read_predictions(args.predictions)
    vals_arr = np.asarray(vals, dtype=float)
    print(f"[aggregate] predictions: {len(vals)}  norm={args.norm}  radius-m={args.radius_m}")

    # Project graph + predictions to a metric CRS.
    Gp = ox.project_graph(G)
    crs = Gp.graph["crs"]
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)],
                        crs="EPSG:4326").to_crs(crs)
    px = pts.geometry.x.to_numpy()
    py = pts.geometry.y.to_numpy()
    pred_xy = np.column_stack([px, py])
    tree = cKDTree(pred_xy)

    # ── spatial-radius aggregation per edge ────────────────────────────────
    print(f"[aggregate] spatial-radius aggregation (R={args.radius_m} m)…")
    all_edges = list(G.edges(keys=True))
    per_edge_contributors: dict[tuple, set[int]] = {}
    per_edge_records: dict[tuple, list[tuple]] = defaultdict(list)
    for u, v, k in all_edges:
        edge = (u, v, k)
        sample_pts, line = _edge_sample_points(Gp, edge, step_m=5.0)
        if not sample_pts:
            per_edge_contributors[edge] = set()
            continue
        found: set[int] = set()
        for sx, sy in sample_pts:
            for idx in tree.query_ball_point([sx, sy], r=args.radius_m):
                found.add(idx)
        per_edge_contributors[edge] = found
        # Compute the distance from each contributor to this edge for the sidecar.
        if found and line is not None:
            for idx in found:
                d = float(line.distance(Point(pred_xy[idx])))
                per_edge_records[edge].append((ids[idx], float(vals[idx]), d))
        elif found:
            # endpoint-fallback case — use Euclidean to the midpoint
            midx, midy = sample_pts[len(sample_pts) // 2]
            for idx in found:
                d = float(np.hypot(pred_xy[idx, 0] - midx, pred_xy[idx, 1] - midy))
                per_edge_records[edge].append((ids[idx], float(vals[idx]), d))

    total_contribs = sum(len(s) for s in per_edge_contributors.values())
    print(f"[aggregate] edge contributions (with duplication across edges): {total_contribs}")

    # ── identify blocks ────────────────────────────────────────────────────
    print("[aggregate] identifying blocks…")
    block_of_edge, blocks = identify_blocks(G)
    avg_edges_per_block = sum(len(es) for es in blocks.values()) / max(1, len(blocks))
    print(f"[aggregate] blocks: {len(blocks)} (avg {avg_edges_per_block:.1f} edges/block)")

    # ── per-edge direct stats (median, p25) ────────────────────────────────
    direct_med_edge: dict[tuple, float] = {}
    direct_p25_edge: dict[tuple, float] = {}
    n_per_edge: dict[tuple, int] = {}
    for edge, idxs in per_edge_contributors.items():
        if not idxs:
            continue
        edge_vals = vals_arr[list(idxs)]
        direct_med_edge[edge] = float(np.median(edge_vals))
        direct_p25_edge[edge] = float(np.percentile(edge_vals, 25))
        n_per_edge[edge] = len(idxs)

    edge_set = [tuple(e) for e in all_edges]
    edge_names = {(u, v, k): _normalize_name(G.get_edge_data(u, v, k).get("name"))
                  for u, v, k in edge_set}

    # neighbour-of-edge: edges sharing an endpoint
    incident: dict[int, set[tuple]] = defaultdict(set)
    for u, v, k in edge_set:
        incident[u].add((u, v, k))
        incident[v].add((u, v, k))
    edge_neighbours = {
        (u, v, k): {e for e in incident[u] | incident[v] if e != (u, v, k)}
        for u, v, k in edge_set
    }

    raw_med_edge, source_edge, counts_edge_med = _smooth_and_fill(
        direct_med_edge, edge_set, edge_neighbours, edge_names)
    raw_p25_edge, _, _ = _smooth_and_fill(
        direct_p25_edge, edge_set, edge_neighbours, edge_names)

    # ── per-block direct stats (median, p25) ───────────────────────────────
    direct_med_block: dict[int, float] = {}
    direct_p25_block: dict[int, float] = {}
    n_per_block: dict[int, int] = {}
    for block_id, block_edges in blocks.items():
        # union of contributors across this block's edges
        all_idxs: set[int] = set()
        for e in block_edges:
            all_idxs.update(per_edge_contributors.get(e, ()))
        if all_idxs:
            block_vals = vals_arr[list(all_idxs)]
            direct_med_block[block_id] = float(np.median(block_vals))
            direct_p25_block[block_id] = float(np.percentile(block_vals, 25))
            n_per_block[block_id] = len(all_idxs)

    # block adjacency + names
    node_to_blocks: dict[int, set[int]] = defaultdict(set)
    block_name_of: dict[int, str | None] = {}
    for bid, b_edges in blocks.items():
        for u, v, k in b_edges:
            node_to_blocks[u].add(bid)
            node_to_blocks[v].add(bid)
            if bid not in block_name_of:
                block_name_of[bid] = edge_names.get((u, v, k))
    block_neighbours: dict[int, set[int]] = defaultdict(set)
    for nbs in node_to_blocks.values():
        for a in nbs:
            for b in nbs:
                if a != b:
                    block_neighbours[a].add(b)

    raw_med_block, source_block, counts_block_med = _smooth_and_fill(
        direct_med_block, list(blocks.keys()), block_neighbours, block_name_of)
    raw_p25_block, _, _ = _smooth_and_fill(
        direct_p25_block, list(blocks.keys()), block_neighbours, block_name_of)

    print(f"[aggregate] edge coverage:  direct={len(direct_med_edge)}  "
          f"smoothed_same_name={counts_edge_med['smoothed_same_name']}  "
          f"smoothed_other={counts_edge_med['smoothed_other']}  "
          f"fallback={counts_edge_med['fallback']}  "
          f"total={len(edge_set)}  "
          f"direct_pct={100*len(direct_med_edge)/max(1,len(edge_set)):.1f}%")
    print(f"[aggregate] block coverage: direct={len(direct_med_block)}  "
          f"smoothed_same_name={counts_block_med['smoothed_same_name']}  "
          f"smoothed_other={counts_block_med['smoothed_other']}  "
          f"fallback={counts_block_med['fallback']}  "
          f"total={len(blocks)}  "
          f"direct_pct={100*len(direct_med_block)/max(1,len(blocks)):.1f}%")
    if n_per_edge:
        n_arr = np.array(list(n_per_edge.values()))
        print(f"[aggregate] direct edges: contributors per edge median={int(np.median(n_arr))} "
              f"p90={int(np.percentile(n_arr, 90))} max={int(n_arr.max())}")
    if n_per_block:
        n_arr = np.array(list(n_per_block.values()))
        print(f"[aggregate] direct blocks: contributors per block median={int(np.median(n_arr))} "
              f"p90={int(np.percentile(n_arr, 90))} max={int(n_arr.max())}")

    # ── normalise each of the 4 populations independently ─────────────────
    edge_order = edge_set
    block_order = list(blocks.keys())

    e_med_scores, info1 = _normalize([raw_med_edge[e] for e in edge_order], args.norm)
    e_p25_scores, info2 = _normalize([raw_p25_edge[e] for e in edge_order], args.norm)
    b_med_scores, info3 = _normalize([raw_med_block[b] for b in block_order], args.norm)
    b_p25_scores, info4 = _normalize([raw_p25_block[b] for b in block_order], args.norm)
    print(f"[aggregate] norm ({args.norm}):  edge_median  {info1}")
    print(f"[aggregate] norm ({args.norm}):  edge_p25     {info2}")
    print(f"[aggregate] norm ({args.norm}):  block_median {info3}")
    print(f"[aggregate] norm ({args.norm}):  block_p25    {info4}")

    b_med_score_of = dict(zip(block_order, b_med_scores))
    b_p25_score_of = dict(zip(block_order, b_p25_scores))

    # ── write edge_lighting.csv ────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "u", "v", "key", "block_id",
            # per-edge median
            "lighting_raw_edge", "lighting_score_edge",
            # per-edge dark (p25)
            "lighting_dark_raw_edge", "lighting_dark_score_edge",
            "n_predictions_edge", "source_edge",
            # per-block median
            "lighting_raw_block", "lighting_score_block",
            # per-block dark (p25)
            "lighting_dark_raw_block", "lighting_dark_score_block",
            "n_predictions_block", "source_block",
        ])
        for edge, e_med_s, e_p25_s in zip(edge_order, e_med_scores, e_p25_scores):
            u, v, k = edge
            bid = block_of_edge.get(edge, -1)
            w.writerow([
                u, v, k, bid,
                f"{raw_med_edge[edge]:.4f}", f"{float(e_med_s):.4f}",
                f"{raw_p25_edge[edge]:.4f}", f"{float(e_p25_s):.4f}",
                n_per_edge.get(edge, 0), source_edge[edge],
                f"{raw_med_block.get(bid, 0.0):.4f}",
                f"{float(b_med_score_of.get(bid, 0.5)):.4f}",
                f"{raw_p25_block.get(bid, 0.0):.4f}",
                f"{float(b_p25_score_of.get(bid, 0.5)):.4f}",
                n_per_block.get(bid, 0), source_block.get(bid, "fallback"),
            ])
    print(f"[aggregate] wrote {len(edge_order)} edges → {args.out}")

    # ── sidecar: image_id → edge mapping ───────────────────────────────────
    # column rename from snap_distance_m to edge_distance_m (semantic shift)
    args.edge_preds_out.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    with args.edge_preds_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "key", "image_id", "raw_pred", "edge_distance_m"])
        for edge in sorted(per_edge_records.keys()):
            u, v, k = edge
            for image_id, raw_pred, ed in per_edge_records[edge]:
                w.writerow([u, v, k, image_id, f"{raw_pred:.4f}", f"{ed:.2f}"])
                n_records += 1
    print(f"[aggregate] wrote {n_records} per-prediction records → {args.edge_preds_out}")


if __name__ == "__main__":
    main()
