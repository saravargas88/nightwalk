"""build_graph.py — download and cache the OSMnx walking graph for the demo area.

Bounding box defaults cover roughly 1 km around Washington Square Park, matching the
geographic spread of the NightWalk dataset (Section 3.1 of the report). Run once;
the downstream scripts read the cached graphml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEMO_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = DEMO_DATA / "wsp_walk.graphml"

# Washington Square Park / Greenwich Village. (south, west, north, east)
# Sized to cover the full geographic extent of efficientnet_train_images.csv
# (lat 40.72688 → 40.73506, lon -74.00328 → -73.99140) with ~30 m buffer so
# every image in the dataset snaps inside the graph.
DEFAULT_BBOX = (40.7260, -74.0040, 40.7355, -73.9910)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=str, default=None,
                    help="south,west,north,east lat/lon (overrides default WSP area)")
    ap.add_argument("--network-type", default="walk", choices=["walk", "drive", "bike", "all"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    DEMO_DATA.mkdir(parents=True, exist_ok=True)

    if args.out.exists() and not args.force:
        print(f"[build_graph] cached graph exists at {args.out}; pass --force to refetch.")
        return

    import osmnx as ox

    if args.bbox:
        south, west, north, east = (float(x) for x in args.bbox.split(","))
    else:
        south, west, north, east = DEFAULT_BBOX

    print(f"[build_graph] downloading {args.network_type} graph for bbox=({south},{west},{north},{east})")
    # osmnx ≥2 uses bbox=(left, bottom, right, top) = (west, south, east, north)
    try:
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=args.network_type)
    except TypeError:
        # fall back to legacy (≤1.x) signature
        G = ox.graph_from_bbox(north, south, east, west, network_type=args.network_type)

    print(f"[build_graph] nodes={len(G.nodes)} edges={len(G.edges)}")
    ox.save_graphml(G, args.out)
    print(f"[build_graph] saved → {args.out}")


if __name__ == "__main__":
    main()
