# Brightness Target Notes

This note explains the current brightness targets in the NightWalk experiments,
which ones are normalized, and which ones are most worth trying next given the
noise in the day-to-night prediction setup.

## Why Target Choice Matters

Right now the problem is not just model capacity. The target itself is noisy.

A day image is only an indirect proxy for the night scene. On top of that, the
night photo brightness depends on:

- camera auto-exposure
- motion blur
- glare and bloom from lampposts
- framing differences between the day and night image
- sky / pavement occupying a large fraction of the image

That means a raw scalar brightness score can be unstable even when the scene
"feels" similarly illuminated.

Because of this, changing the target metric is one of the highest-value things
to test.

## Current Brightness Targets

These targets are computed in
[`run_brightness_metric_experiments.py`](/Users/sara/Desktop/SPRING2026/CV/NightWalk/brightnessmetricexperiments/run_brightness_metric_experiments.py)
and stored in
[`paired_dataset_with_brightness.csv`](/Users/sara/Desktop/SPRING2026/CV/NightWalk/brightnessmetricexperiments/experiment_outputs/paired_dataset_with_brightness.csv).

### Raw brightness-style targets

- `gray_mean`
  Mean grayscale brightness over the whole night image.
  Simple baseline, but very sensitive to dark pavement, sky, and exposure.

- `gray_median`
  Median grayscale brightness.
  More robust to a few very bright pixels, but may ignore the lamppost signal too much.

- `gray_trimmed_mean`
  Mean grayscale after trimming the darkest and brightest tails.
  Useful when extreme glare or extreme darkness adds noise.

- `gray_p90`
  The 90th percentile grayscale value.
  More highlight-sensitive than `gray_mean`; can help if lampposts and storefronts
  are the key signal.

- `luma_mean`
  Mean perceptual luminance using RGB weighting.
  Similar to grayscale mean, but slightly more perceptually grounded.

- `value_mean`
  Mean HSV-like "value" channel, driven by the brightest color channel.
  Often more sensitive to visible light sources than plain grayscale.

### Normalized targets

- `gray_mean_over_std`
  Per-image normalization:
  whole-image mean divided by that same image's brightness standard deviation.
  This tries to capture brightness relative to local contrast.

- `gray_mean_zscore`
  Dataset-level normalization:
  `(gray_mean - dataset_mean) / dataset_std`
  This does not change the rank ordering much, but it standardizes the scale.

- `gray_mean_robust_zscore`
  Robust dataset-level normalization:
  `(gray_mean - median) / MAD-like scale`
  Less sensitive to outliers than the ordinary z-score.

## Which Targets Are Most Promising

Given the current noise and the lamppost problem, I would not prioritize all
targets equally.

### Best first targets to try

- `gray_mean_zscore`
  Best first normalized target.
  Good when exposure variation is part of the noise, and a stable scale helps.

- `value_mean`
  Best raw target to try if we want more sensitivity to actual visible light sources.
  This is especially relevant for lampposts and bright storefronts.

- `gray_p90`
  Strong candidate when the useful signal lives in the bright tail rather than
  the whole-image average.

### Second-tier targets

- `luma_mean`
  Worth trying, but usually not as targeted to the lamppost issue as `value_mean`
  or `gray_p90`.

- `gray_trimmed_mean`
  Worth trying if glare and saturated hotspots are overwhelming the regression.

### Lower-priority targets

- `gray_mean`
  Good baseline, but probably too sensitive to irrelevant dark or bright regions.

- `gray_median`
  Often too insensitive to sparse bright sources like lampposts.

- `gray_mean_over_std`
  Interesting exploratory target, but it was not especially strong in the linear
  experiments.

## Recommendation For The Lamppost Issue

The lamppost issue is not only that many rows have zero lamppost count. It is
also that a single scalar like `gray_mean` may not reflect lamppost-driven
illumination very well.

A few implications:

- `value_mean` and `gray_p90` are more likely than `gray_mean` to react to bright,
  sparse light sources.
- Binned classification may work better than raw regression because it reduces
  sensitivity to exact exposure noise.
- Using a normalized target like `gray_mean_zscore` helps when the main issue is
  inconsistent scale across night images.

So if the goal is to improve the lamppost-related signal, the first things I
would test are:

1. `gray_mean_zscore` with 4 bins
2. `value_mean` with 4 bins
3. `gray_p90` with 4 bins
4. If these are still noisy, retry with 3 bins instead of 4

## Suggested Training Order

### Classification first

For the current EfficientNet brightness-level pipeline, the first pass I would run is:

```bash
python3 /Users/sara/Desktop/SPRING2026/CV/NightWalk/model-training/train_efficientnet_brightness_levels.py --metric gray_mean_zscore --bins 4
python3 /Users/sara/Desktop/SPRING2026/CV/NightWalk/model-training/train_efficientnet_brightness_levels.py --metric value_mean --bins 4
python3 /Users/sara/Desktop/SPRING2026/CV/NightWalk/model-training/train_efficientnet_brightness_levels.py --metric gray_p90 --bins 4
```

If 4 bins is too unstable:

```bash
python3 /Users/sara/Desktop/SPRING2026/CV/NightWalk/model-training/train_efficientnet_brightness_levels.py --metric gray_mean_zscore --bins 3
```

### Regression second

If one metric looks meaningfully learnable in classification, that is the best
candidate to revisit with a regression head later.

## Multiple-Metric Training

The longer-term idea still makes sense:

- one shared EfficientNet backbone
- multiple output heads or multiple output channels
- one target per brightness definition

That is useful for asking:
"Which notion of brightness is most learnable from daytime imagery?"

But because local training on MPS is likely to be slow, the best workflow is:

1. start with one metric at a time
2. identify the most learnable target
3. only then consider multi-target training

That keeps the experiment loop cheaper and easier to interpret.
