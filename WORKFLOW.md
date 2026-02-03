# Workflow (Repo Conventions)

## 1) Directory roles
- `docs/`: human-written notes (charter, research log, personal notes).
- `configs/`: experiment definitions (provider/model/params).
- `src/` + `scripts/`: project code and entry scripts.
- `reproduction/`: reproduction of official EDINET-Bench baselines.
  - `reproduction/sampling/`: frozen sample specs and ID lists.
  - `reproduction/results/`: small summaries (tables/metrics/CI).
  - `reproduction/outputs/`: raw run artifacts.
- `external/EDINET-Bench/`: upstream EDINET-Bench repo (submodule).
- `outputs/`: raw run artifacts for non-reproduction experiments.
- `data/`: datasets/caches.

## 2) Config vs secrets
- `.env` is for keys/endpoints only.
- Experiment settings (model choice, decoding params, etc.) live in `configs/`.

## 3) Run bookkeeping
- Assign an ID to each meaningful run: `EXP-0001`, `EXP-0002`, ...
- Store raw artifacts under `outputs/<EXP-id>/` (or `reproduction/outputs/<EXP-id>/`).
- Record each run in `docs/research_log.md`: config path + output dir + short conclusion.
