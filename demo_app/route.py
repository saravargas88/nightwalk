"""route.py — compute shortest-path and well-lit walking routes.

Edge cost model (Section 5.2):

    shortest_cost = length_m
    lit_cost      = length_m · (1 + λ·(1 − lighting_score)) · (1 + μ_unk·is_fallback)

`lighting_score` ∈ [0, 1] comes from aggregate_to_edges.py. The CSV holds two
parallel aggregation modes (per-edge / per-block) AND two metrics (median /
dark p25) — caller picks via `load_edge_scores(path, mode, metric)`.

At lambda_=0 and μ_unk=0 the two routes coincide. Larger λ trades extra
distance for higher lighting. Larger μ_unk pushes routes away from
fallback (unknown / inferred) units.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEMO_DATA = Path(__file__).resolve().parent / "data"


@dataclass
class RouteResult:
    label: str
    nodes: list[int]
    coords: list[tuple[float, float]]  # (lat, lon) in order
    distance_m: float
    mean_lighting: float
    min_lighting: float


# ── loaders ────────────────────────────────────────────────────────────────
def _score_column(mode: str, metric: str) -> str:
    if mode not in ("edge", "block"):
        raise ValueError(f"mode must be 'edge' or 'block', got {mode!r}")
    if metric not in ("median", "dark"):
        raise ValueError(f"metric must be 'median' or 'dark', got {metric!r}")
    if metric == "median":
        return f"lighting_score_{mode}"
    return f"lighting_dark_score_{mode}"


def load_edge_scores(path: Path, mode: str = "edge",
                     metric: str = "median") -> dict[tuple[int, int, int], float]:
    """Read just the chosen mode×metric score column."""
    col = _score_column(mode, metric)
    scores: dict[tuple[int, int, int], float] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["u"]), int(r["v"]), int(r["key"]))
            scores[key] = float(r[col])
    return scores


def load_edge_sources(path: Path, mode: str = "edge") -> dict[tuple[int, int, int], str]:
    """Read the source flag for the chosen aggregation mode."""
    if mode not in ("edge", "block"):
        raise ValueError(f"mode must be 'edge' or 'block', got {mode!r}")
    col = f"source_{mode}"
    out: dict[tuple[int, int, int], str] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["u"]), int(r["v"]), int(r["key"]))
            out[key] = r[col]
    return out


def load_edge_table(path: Path) -> dict[tuple[int, int, int], dict]:
    """Full edge_lighting.csv row keyed by (u, v, key) — both modes × both metrics."""
    out: dict[tuple[int, int, int], dict] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["u"]), int(r["v"]), int(r["key"]))
            out[key] = {
                "block_id": int(r["block_id"]),
                # per-edge × median
                "lighting_raw_edge":      float(r["lighting_raw_edge"]),
                "lighting_score_edge":    float(r["lighting_score_edge"]),
                # per-edge × dark (p25)
                "lighting_dark_raw_edge": float(r["lighting_dark_raw_edge"]),
                "lighting_dark_score_edge": float(r["lighting_dark_score_edge"]),
                "n_predictions_edge":     int(r["n_predictions_edge"]),
                "source_edge":            r["source_edge"],
                # per-block × median
                "lighting_raw_block":     float(r["lighting_raw_block"]),
                "lighting_score_block":   float(r["lighting_score_block"]),
                # per-block × dark (p25)
                "lighting_dark_raw_block": float(r["lighting_dark_raw_block"]),
                "lighting_dark_score_block": float(r["lighting_dark_score_block"]),
                "n_predictions_block":    int(r["n_predictions_block"]),
                "source_block":           r["source_block"],
            }
    return out


def load_edge_predictions(path: Path) -> dict[tuple[int, int, int], list[dict]]:
    """Sidecar: per-prediction records keyed by edge. Distance column renamed
    `edge_distance_m` in v3 (was `snap_distance_m`); we accept both for back-compat.
    """
    out: dict[tuple[int, int, int], list[dict]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["u"]), int(r["v"]), int(r["key"]))
            dist_key = "edge_distance_m" if "edge_distance_m" in r else "snap_distance_m"
            out.setdefault(key, []).append({
                "image_id": r["image_id"],
                "raw_pred": float(r["raw_pred"]),
                "edge_distance_m": float(r[dist_key]),
            })
    return out


# ── routing ────────────────────────────────────────────────────────────────
def annotate_graph(G, edge_scores: dict, lambda_: float,
                   unknown_penalty: float = 0.0,
                   edge_sources: dict | None = None):
    """Mutates G: writes a `lit_cost` to every edge.

    Args:
      lambda_:        lighting weight. lit_cost = length·(1 + λ·(1−score))·…
      unknown_penalty (μ_unk): extra multiplicative cost applied to edges whose
                      source is "fallback". 0 disables. Default 0.
      edge_sources:   optional dict[(u,v,k)] → source string. If omitted, the
                      penalty is inactive.
    """
    import math

    sources = edge_sources or {}
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 0.0))
        score = edge_scores.get((u, v, k), 0.5)
        cost = length * (1.0 + lambda_ * (1.0 - score))
        is_fallback = sources.get((u, v, k)) == "fallback"
        if is_fallback and unknown_penalty:
            cost *= (1.0 + unknown_penalty)
        data["lit_cost"] = cost
        data["lighting_score"] = score
        if not math.isfinite(data["lit_cost"]):
            data["lit_cost"] = length


def _path_stats(G, nodes: list[int], edge_scores: dict, label: str) -> RouteResult:
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes]
    distance = 0.0
    lightings: list[float] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        edges = G.get_edge_data(u, v) or {}
        if not edges:
            continue
        k = min(edges, key=lambda kk: edges[kk].get("length", float("inf")))
        edata = edges[k]
        distance += float(edata.get("length", 0.0))
        lightings.append(float(edge_scores.get((u, v, k), edge_scores.get((v, u, k), 0.5))))
    mean_l = sum(lightings) / len(lightings) if lightings else 0.0
    min_l = min(lightings) if lightings else 0.0
    return RouteResult(label=label, nodes=nodes, coords=coords,
                       distance_m=distance, mean_lighting=mean_l, min_lighting=min_l)


def compute_routes(G, edge_scores: dict, start_xy: tuple[float, float],
                   end_xy: tuple[float, float], lambda_: float = 2.0,
                   unknown_penalty: float = 0.0,
                   edge_sources: dict | None = None) -> tuple[RouteResult, RouteResult]:
    """Return (shortest_route, lit_route). start_xy and end_xy are (lat, lon)."""
    import networkx as nx
    import osmnx as ox

    annotate_graph(G, edge_scores, lambda_=lambda_,
                   unknown_penalty=unknown_penalty, edge_sources=edge_sources)

    start_lat, start_lon = start_xy
    end_lat, end_lon = end_xy
    src = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    dst = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

    shortest_nodes = nx.shortest_path(G, src, dst, weight="length")
    lit_nodes = nx.shortest_path(G, src, dst, weight="lit_cost")

    shortest = _path_stats(G, shortest_nodes, edge_scores, "shortest")
    lit = _path_stats(G, lit_nodes, edge_scores, "well-lit")
    return shortest, lit


def summary_lines(shortest: RouteResult, lit: RouteResult) -> list[str]:
    detour = (lit.distance_m / shortest.distance_m) if shortest.distance_m else float("nan")
    return [
        f"shortest : {shortest.distance_m:6.0f} m   mean_lighting={shortest.mean_lighting:.3f}",
        f"well-lit : {lit.distance_m:6.0f} m   mean_lighting={lit.mean_lighting:.3f}   "
        f"detour={detour:.2f}×",
    ]
