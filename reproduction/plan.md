# Reproduction Plan

Scope: fraud detection + earnings forecasting (industry: skip).

## Data
- Use HF dataset `SakanaAI/EDINET-Bench`.
- Follow dataset usage restriction (do not target/harm real companies).

## Per-run artifacts (applies to every reproduction run)
- Assign an `EXP-R-xxxx` id.
- Raw artifacts → `reproduction/outputs/<EXP-R-xxxx>/`
  (logs, run metadata, raw predictions, intermediate files).
- Small summaries → `reproduction/results/`
  - `<EXP-R-xxxx>_summary.md` (short run note)
  - task summaries (metrics/CI JSONs) as needed.

## Steps
1) Run official evaluation end-to-end on a tiny slice (sanity).
2) Record minimal pipeline notes (code pointers):
   - TSV fields used (BS/PL/CF/Summary/Text, etc.)
   - prompt outline + concatenation order
   - output parsing → label/score
   - metrics computation + aggregation protocol
   Write to `reproduction/notes.md`.
3) Budgeted reproduction:
   - define stratified sampling (fraud by label; earnings by target bins or proxy)
   - freeze ID lists under `reproduction/sampling/`
   - run eval restricted to frozen IDs
   - compute point estimates + bootstrap CIs; write to `reproduction/results/`
