# Reproducibility checklist

**Audit date:** 2026-10-11

| Item | Present? | Path / notes |
|---|---|---|
| README | Yes (rewritten Version B) | `README.md` |
| LICENSE | Yes | `LICENSE` (MIT) |
| requirements.txt | Yes (pinned) | `requirements.txt` |
| environment.yml | Yes | `environment.yml` |
| Exact Python version | Yes | 3.13.2 (`paper/tables/environment.csv`) |
| Exact package versions | Yes (partial) | torch 2.7.1, torchvision 0.22.1, timm 1.0.29, … |
| Exact seed list | Yes | Primary 42–47; secondary noted in tables |
| Configuration files | Yes | `configs/`, `experiments/*/config.yaml` |
| Training script | Yes | `src/train.py` |
| Evaluation script | Yes | `src/evaluate.py` / eval sweeps in `scripts/` |
| Preprocessing script | Yes | `src/preprocessing.py`, cache builders |
| Duplicate-audit script | Yes | `scripts/audit_aptos_split_duplicates.py` |
| Calibration / temperature | Yes | metrics + ECE tables |
| Metric scripts | Yes | `src/metrics.py`, table builders |
| Table-generation scripts | Yes | `scripts/build_manuscript_md.py`, `build_paper_v3_tables.py`, … |
| Figure-generation scripts | Yes | `paper/figures/*`, `scripts/build_*figures*` |
| Per-seed metrics | Yes | `paper/tables/master_results_per_run.csv`, results JSON |
| Per-image predictions | Yes | `results/*/…_predictions.json` (local) |
| Per-image logits | Present for many runs | Confirm licence before public upload |
| Run logs | Yes | `logs/`, experiment folders |
| Checkpoint metadata | Yes | `checkpoints/`, `best.pt` locally — **large; optional for Zenodo** |
| Protocol-deviation docs | Yes | `docs/protocol_deviation.md` |
| Threshold-head audit docs | Yes | `docs/threshold_head_audit.md` |
| Zenodo metadata | Yes | `.zenodo.json`, `CITATION.cff` |
| GitHub URL | Yes (private until release) | https://github.com/Zubair121Md/DR_RP in `CITATION.cff` / `.zenodo.json`; must be made public before Zenodo archiving |
| Zenodo DOI | **TODO** | After release |
| APTOS partition archive URL/hash | Yes | `data_manifest/aptos_archive_hash.csv`, `DATASET_PROVENANCE.md` |
| NLL/Brier (H1/H2) | Yes | `paper/tables/deduplication_sensitivity*.csv`, Supplementary Table S6 |

## Hard do-nots for release

- Do not upload APTOS/IDRiD/Messidor-2/DeepDRiD **images**.
- Do not claim six exchangeable seeds.
- Do not claim threshold-head calibration.
