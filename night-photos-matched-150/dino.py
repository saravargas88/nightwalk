"""dino_exps.py
Grounding DINO experiment runner.
Prompts are defined in prompts.yaml — add new experiments there without
touching this file.

Modes:
  - grid:   runs all prompts x all thresholds on N_IMAGES sample images,
            saves annotated bounding box images and a summary CSV.
  - count:  runs a single named prompt on SAMPLES daytime images,
            saves dino_counts_{RUN_NAME}.csv and dino_counts_{RUN_NAME}.json
            to dino_counts/ for use in proxy_viewer.html.

Set MODE below to switch between them.
"""
import csv
import re
import json
import yaml
import sys
import torch
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import DATA_DIR, CSV_PATH

# ── Config ────────────────────────────────────────────────────────────────────
MODE = "count"   # "grid" or "count"

MODEL_ID = "IDEA-Research/grounding-dino-base"

# Grid mode settings
N_IMAGES = 20
GRID_OUT = Path("dino_grid")

THRESHOLDS = [
    {"name": "high",   "box": 0.40, "text": 0.35},
    {"name": "medium", "box": 0.30, "text": 0.25},
    {"name": "low",    "box": 0.20, "text": 0.15},
]

# Count mode settings
COUNT_PROMPT_NAME = "informed_prompt_3"  # must match a name in prompts.yaml
SAMPLES           = 150
THRESHOLD         = 0.30
TEXT_THRESHOLD    = 0.25
COUNTS_DIR        = Path("dino_counts")
COUNTS_DIR.mkdir(exist_ok=True)

# Bounding box colors per category
COLORS = {
    "tree":        "#4CAF50",
    "storefront":  "#2196F3",
    "lamppost":    "#FF9800",
    "streetlight": "#FF9800",
    "doorman":     "#9C27B0",
    "other":       "#999999",
}

# ── Load prompts from YAML ────────────────────────────────────────────────────
_PROMPTS_YAML = Path(__file__).parent.parent / "dino_experiments" / "prompts.yaml"
with open(_PROMPTS_YAML) as f:
    _config = yaml.safe_load(f)

PROMPTS = [
    {
        "name":     p["name"],
        "text":     p["text"],
        "patterns": {k: re.compile(v) for k, v in p["patterns"].items()}
    }
    for p in _config["prompts"]
]

PROMPT_MAP = {p["name"]: p for p in PROMPTS}

# ── Load dataset ──────────────────────────────────────────────────────────────
_PAIRS_CSV = Path(__file__).parent / "matches_remapped.csv"
_pairs = pd.read_csv(_PAIRS_CSV)
_pairs = _pairs[_pairs["skipped"] == False].reset_index(drop=True)
day_df = _pairs[["night_photo", "day_image"]].rename(columns={"day_image": "image"})
day_df["taken_on"] = ""
day_df["period"] = ""

# ── Load model ────────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
print(f"Model loaded: {MODEL_ID} on {device}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect(image, text, box_thresh, text_thresh):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.size[::-1]]
    )[0]

def count_detections(detections, patterns):
    counts = {key: 0 for key in patterns}
    label_map = {}
    for label in detections["labels"]:
        label_lower = label.lower()
        matched = "other"
        for key, pattern in patterns.items():
            if pattern.search(label_lower):
                counts[key] += 1
                matched = key
                break
        label_map[label] = matched
    return counts, label_map

def draw_boxes(image, detections, label_map):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for box, label, score in zip(
        detections["boxes"].cpu().numpy(),
        detections["labels"],
        detections["scores"].cpu().numpy()
    ):
        x0, y0, x1, y1 = box
        color = COLORS.get(label_map.get(label, "other"), COLORS["other"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.rectangle([x0, y0, x0 + 120, y0 + 16], fill=color)
        draw.text((x0 + 2, y0 + 1), f"{label} {score:.2f}", fill="white")
    return draw_img

# ─────────────────────────────────────────────────────────────────────────────
# GRID MODE
# ─────────────────────────────────────────────────────────────────────────────
if MODE == "grid":
    sample_rows = []
    for _, row in day_df.iterrows():
        if (DATA_DIR / row["image"]).exists():
            sample_rows.append(row)
        if len(sample_rows) >= N_IMAGES:
            break
    print(f"Loaded {len(sample_rows)} sample images for grid")

    all_results = []

    for prompt in PROMPTS:
        for thresh in THRESHOLDS:
            exp_name = f"{prompt['name']}__{thresh['name']}"
            exp_dir  = GRID_OUT / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n── Experiment: {exp_name} ──────────────────────")

            exp_rows = []
            for row in sample_rows:
                image = Image.open(DATA_DIR / row["image"]).convert("RGB")
                dets  = detect(image, prompt["text"], thresh["box"], thresh["text"])
                counts, label_map = count_detections(dets, prompt["patterns"])

                annotated = draw_boxes(image, dets, label_map)
                img_stem  = Path(row["image"]).stem
                annotated.save(exp_dir / f"{img_stem}.jpg")

                result = {
                    "experiment":  exp_name,
                    "prompt":      prompt["name"],
                    "threshold":   thresh["name"],
                    "box_thresh":  thresh["box"],
                    "text_thresh": thresh["text"],
                    "image":       row["image"],
                    **{k: v for k, v in counts.items()},
                    "total":       sum(counts.values()),
                }
                exp_rows.append(result)
                print(f"  {Path(row['image']).name[-40:]}  " +
                      "  ".join(f"{k}={v}" for k, v in counts.items()))

            exp_csv = exp_dir / "counts.csv"
            with open(exp_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=exp_rows[0].keys())
                w.writeheader()
                w.writerows(exp_rows)

            all_results.extend(exp_rows)

    summary_path = GRID_OUT / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved summary to {summary_path}")

    summary_df   = pd.DataFrame(all_results)
    numeric_cols = [c for c in summary_df.columns if summary_df[c].dtype in ["int64", "float64"]]
    agg = summary_df.groupby("experiment")[numeric_cols].mean().round(2)
    print(f"\nMean counts per image:\n{agg.to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# COUNT MODE
# ─────────────────────────────────────────────────────────────────────────────
elif MODE == "count":
    if COUNT_PROMPT_NAME not in PROMPT_MAP:
        raise ValueError(f"Prompt '{COUNT_PROMPT_NAME}' not found in prompts.yaml")

    prompt      = PROMPT_MAP[COUNT_PROMPT_NAME]
    RUN_NAME    = COUNT_PROMPT_NAME
    OUTPUT_CSV  = COUNTS_DIR / f"dino_counts_{RUN_NAME}-pairs.csv"
    OUTPUT_JSON = COUNTS_DIR / f"dino_counts_{RUN_NAME}-pairs.json"

    print(f"Prompt:     {COUNT_PROMPT_NAME}")
    print(f"Text:       {prompt['text']}")
    print(f"Thresholds: box={THRESHOLD}  text={TEXT_THRESHOLD}")
    print(f"Samples:    {SAMPLES}")
    print(f"Output:     {OUTPUT_CSV}")
    print("-" * 60)

    day_rows    = day_df.to_dict("records")
    output_rows = []
    box_data    = {}
    processed   = 0

    for row in tqdm(day_rows, total=SAMPLES, desc="Processing", unit="img"):
        if processed >= SAMPLES:
            break

        img_path = DATA_DIR / row["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"Skipping {img_path}: {e}")
            continue

        dets = detect(image, prompt["text"], THRESHOLD, TEXT_THRESHOLD)
        counts, label_map = count_detections(dets, prompt["patterns"])

        box_data[row["image"]] = [
            {
                "box":      box,
                "label":    label,
                "score":    round(score, 3),
                "category": label_map.get(label, "other"),
            }
            for box, label, score in zip(
                dets["boxes"].cpu().numpy().tolist(),
                dets["labels"],
                dets["scores"].cpu().numpy().tolist()
            )
        ]

        output_rows.append({
            "night_photo": row["night_photo"],
            "image":       row["image"],
            "taken_on":    row["taken_on"],
            "period":      row["period"],
            **counts,
            "run":         RUN_NAME,
        })

        processed += 1
        tqdm.write(
            f"[{processed}/{SAMPLES}] {Path(row['image']).name[-40:]}  " +
            "  ".join(f"{k}={v}" for k, v in counts.items())
        )

    fieldnames = ["night_photo", "image", "taken_on", "period"] + list(prompt["patterns"].keys()) + ["run"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(box_data, f)

    print(f"\nSaved {len(output_rows)} rows to {OUTPUT_CSV}")
    print(f"Saved box data to {OUTPUT_JSON}")

else:
    raise ValueError(f"Unknown MODE: {MODE!r}. Use 'grid' or 'count'.")