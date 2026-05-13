# NightWalk demo app — lighting-aware route planning

Implementation of §5 of the NightWalk report — *"Application: Safer Route
Planning"*. Takes the per-image nighttime-brightness predictions produced by
the trained model and turns them into a routable lighting map over Greenwich
Village, then computes and visualises **well-lit** pedestrian routes
alongside the **shortest** ones for comparison.

The model's daytime input comes from the original fleet-vehicle imagery
(cameras pointed at the sidewalks on either side of the car, per the
dataset's collection setup — see report §3.1), so each prediction reflects
what a pedestrian would see at street level. Predictions are pooled spatially
onto the OSM walking-network and used as edge weights in a length-vs-lighting
trade-off (controlled by λ).

## Pipeline

```
predictions.csv          → aggregate_to_edges.py → edge_lighting.csv
(image_id, lat, lon,                              + edge_predictions.csv
 heading, brightness)                                          │
            ▲                                                  ▼
            │                                                route.py
   build_predictions.py                                        │
   (real model OR --mock)                                      ▼
                                                  render_map.py   app.py
                                                  (static HTML)   (Streamlit)
```

The five layers are decoupled. The input contract is the columns of
`data/predictions.csv` — anything that writes that file feeds the rest unchanged.

### `data/edge_lighting.csv` (v3 schema)

One row per OSM walking-graph edge, with **four parallel score columns**
(median × {edge, block} and dark/p25 × {edge, block}) plus `block_id`:

```
u, v, key, block_id,
lighting_raw_edge, lighting_score_edge,
lighting_dark_raw_edge, lighting_dark_score_edge,
n_predictions_edge, source_edge,
lighting_raw_block, lighting_score_block,
lighting_dark_raw_block, lighting_dark_score_block,
n_predictions_block, source_block
```

Edges in the same block share identical `*_block` values; `*_edge` differ.
`source_*` ∈ `{direct, smoothed_same_name, smoothed_other, fallback}`.

### `data/edge_predictions.csv`

One row per (edge, contributing prediction) pair:
`u, v, key, image_id, raw_pred, edge_distance_m`. Used by the inspector + the
thumbnail strip to trace every score back to the daytime images that produced it.

## Setup

From repo root:

```bash
pip install -r requirements.txt
```

The demo adds `osmnx`, `networkx`, `folium`, `geopandas`, `shapely`, `scipy`,
`streamlit`, and `streamlit-folium` on top of the training requirements.

## Run end-to-end with the real model

```bash
python demo_app/build_graph.py --force          # OSMnx walk graph, cached
python demo_app/build_predictions.py            # real model over urban-mosaic
python demo_app/aggregate_to_edges.py           # spatial-radius aggregation
python demo_app/render_map.py --examples        # → grid + layered HTMLs
streamlit run demo_app/app.py                   # interactive UI
```

`build_predictions.py` defaults: fold-3 finetune checkpoint, auto-locates
`urban-mosaic/washington-square/` by walking up from the worktree. Override
with `--checkpoint`, `--image-dir`, `--source test_split`, `--limit N`.

After regenerating predictions, re-run `aggregate_to_edges.py` and the renderers.

## Run with mock predictions

```bash
python demo_app/build_graph.py
python demo_app/build_predictions.py --mock
python demo_app/aggregate_to_edges.py
python demo_app/render_map.py --examples
```

Mock produces synthetic but spatially structured values so routing actually
picks different paths.

## Aggregation (§5.1 — v3)

**Spatial-radius pooling.** For each OSM edge, the aggregator pools every
prediction within `--radius-m R` (default **30 m**) of any point along the
edge's geometry — replaces the v2 "nearest-edge snap" which left most of the
fine-grained walking-graph edges empty. Expected coverage at the WSP graph:

| | direct | smoothed | fallback |
|---|---|---|---|
| v2 (snap, 20 m) | 41 % | 33 % | 26 % |
| v3 (spatial-radius, 30 m) | **82 %** | 10 % | 8 % |

Each edge gets both a **median** (typical lighting) and a **p25** (the
"darkest-quartile" lighting) score so a route cost can optimise either:
- *Typical* favours routes with high average brightness.
- *Pessimistic (p25)* favours routes whose **darkest stretch** is as bright as
  possible — closer to what a pedestrian-safety user cares about.

**Block aggregation.** Edges are also grouped into *blocks* — named-street
runs between real (degree-≥3) intersections. Each block pools all its
constituent edges' contributors. Block direct-coverage is even higher
(~91 % at v3).

**Smoothing & fallback** are now rare edge cases:
1. *same-name* — empty units inherit from same-OSM-name neighbours first.
2. *other* — any adjacent unit's value.
3. *fallback* — dataset median.

Tag is recorded in the `source_*` column.

## Routing (§5.2)

```
shortest_cost = length_m
lit_cost      = length_m · (1 + λ·(1 − score)) · (1 + μ_unk · is_fallback)
```

`λ` is the lighting weight (UI slider). `μ_unk` is the unknown-area penalty:
edges whose `source_*` is `fallback` cost `(1 + μ_unk)` times more — pushes
routes away from unmeasured streets. Default `μ_unk = 0.5`. `μ_unk = 0`
reproduces v2 routing.

## Static figures — `render_map.py --examples`

Default `--layout both` produces two HTMLs:

- **`route_viewer_grid.html`** — three side-by-side mini-maps, one per
  example pair. Each panel auto-zooms to its own pair and shows per-pair
  stats below the map. Dark basemap tile (`cartodbdark_matter`). This is the
  figure for the report.
- **`route_viewer_layered.html`** — single big map with one `FeatureGroup`
  per pair behind a layer-control widget; only pair 1 visible on first
  paint. For the live demo.

Other CLI flags:

| flag | choices | default |
|---|---|---|
| `--mode` | `edge`, `block` | `edge` |
| `--metric` | `median`, `dark` | `median` |
| `--layout` | `grid`, `layered`, `both` | `both` |
| `--lambda` | float | `2.0` |
| `--mu-unk` | float | `0.5` |

## Streamlit UI (`streamlit run demo_app/app.py`)

Sidebar controls:

- **Aggregation** — `per edge` (default) or `per block`. Drives basemap colours,
  routing, and the audit panel.
- **Lighting metric** — `Typical (median)` or `Pessimistic (p25)`. Picks which
  score column drives routing.
- **Click mode** — `Set endpoints` or `Inspect edge`.
- **λ** slider (0–8) — lighting weight in the cost.
- **μ_unk** slider (0–2) — extra penalty on fallback (unknown) units.

Main area:

- The map (dark basemap, navy→yellow lighting palette, blue + pink-dashed routes).
- A per-segment thumbnail strip below the map when both endpoints are set —
  one card per edge in walking order, with the daytime image whose prediction
  was closest to the per-edge median.
- The **Inspect edge** panel (when click mode is active and an edge has been
  clicked) shows: direct predictions, then aggregations as a 2 × 2 grid
  (median × edge, p25 × edge, median × block, p25 × block) with colour swatches.
- A collapsible **Pipeline audit** with coverage breakdown and score histograms.

### Thumbnail source images

The thumbnail strip walks up from the worktree looking for these
directories, first hit wins:

1. `urban-mosaic/washington-square/` (full-res, original)
2. `nightwalk-images-224/` (resized HPC training set)

If neither is present the card falls back to a text placeholder with the
image filename + raw prediction + edge distance.

## Files

| File | Role |
|---|---|
| `build_predictions.py` | model → predictions.csv (`--mock` for synthetic) |
| `build_graph.py` | OSMnx graph download + graphml cache (full-extent bbox) |
| `aggregate_to_edges.py` | predictions → edge + block lighting scores (spatial-radius + median + p25) |
| `route.py` | shortest + well-lit routing; mode × metric loaders; μ_unk penalty |
| `render_map.py` | static HTML; grid + layered; dark tile; report polish |
| `app.py` | Streamlit interactive UI |
| `data/edge_lighting.csv` | per-edge rows × four score columns + `block_id` |
| `data/edge_predictions.csv` | per-(edge, contributor) sidecar (image → edge map) |
| `data/` | generated artefacts (gitignored) |

## Tuning

- **Geographic scope** — `DEFAULT_BBOX` in `build_graph.py`.
- **Spatial radius** — `aggregate_to_edges.py --radius-m 30`.
- **Normalisation** — `aggregate_to_edges.py --norm {percentile, sigmoid}`
  (percentile default).
- **Default aggregation** — `render_map.py --mode {edge, block}` (edge default).
- **Lighting metric** — `render_map.py --metric {median, dark}` (median default).
- **Lighting weight & unknown penalty** — `--lambda`, `--mu-unk` flags or the
  Streamlit sliders.
- **Example pairs** — edit `EXAMPLE_PAIRS` at the top of `render_map.py`.
