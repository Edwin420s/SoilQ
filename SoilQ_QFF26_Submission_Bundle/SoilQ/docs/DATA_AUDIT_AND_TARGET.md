# Dataset audit and target-selection protocol

Official Kenyan maize dataset: https://data.mendeley.com/datasets/8bx88zpvdp/2

The published description mentions two Kenyan on-station sites across four cropping seasons, with rainfall, soil moisture at crop stages, plant height, chlorophyll, and yields. This does not establish that the downloadable files have daily data or direct irrigation-decision labels.

## Audit checklist
- Record source, authors, DOI/version, license, download date and SHA-256.
- Establish what one row represents: plot, plant, treatment, site-season, or time point.
- Inspect units, depth, sampling dates, crop stage, missing values, duplicates and measurement method.
- Identify site/plot/season/treatment/repetition identifiers and potential train/test leakage.
- Check that target and features refer to the same prediction time.
- Document target definition and evidence before training.

## Target policy
Use a source-provided binary target with a clear definition, or a proxy derived from a documented agronomic threshold appropriate to crop, soil, depth and context. Clearly label proxy targets as proxies.

Do not arbitrarily split values at the median just to force two classes. Do not call yield or treatment categories “irrigation urgency” without evidence. If the Kenyan dataset is sparse or lacks a defensible target, document that and use it as agricultural context; consider EOCIS Africa Soil Moisture for daily time-series exploration while noting it is model-derived, coarse-grid data without farmer decision labels.

## Evaluation
The included stratified random holdout is only a baseline. If repeated rows come from the same plot/site/season, adapt evaluation to use group-aware splits. For time series, use chronological evaluation. Do not allow related observations to leak across train/test.

Report data source, license, hash, target/features, exclusions, split strategy, seed, preprocessing, class counts, metrics, confusion matrix, limitations and whether the model was simulated or run on hardware.
