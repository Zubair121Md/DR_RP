# Phase O / P — optional extensions

## Phase O — CNN vs Hybrid vs Transformer
Matched recipe: **green_clahe × CE**.
- CNN: existing EfficientNet-B0 G+CLAHE×CE (seeds 42/43/44)
- Hybrid: existing MobileViT H2
- Transformer: **Swin-Tiny** (new) → `configs/phase_o/`

```bash
cd "/Users/zubairishaq/Desktop/dr- research"
pkill -f 'phase_o|phase_p|src.train|src.evaluate|src.train_lodo' 2>/dev/null; sleep 2
nohup caffeinate -is bash scripts/keepalive_phase_o.sh >> logs/keepalive_phase_o_outer.log 2>&1 &
echo $! > logs/keepalive_phase_o.pid

# watch
tail -f logs/phase_o_swin.log
# done when: PHASE_O_ALL_OK
# summary: experiments/results/phase_o_arch_comparison.md
```

## Phase P — LODO (leave-one-dataset-out)
Train on APTOS train + all externals **except** held-out; val = APTOS val; eval held-out.
Model: MobileViT green_clahe × CE.

**Seed 42:** done (`PHASE_P_ALL_OK` in `logs/phase_p_lodo_s42.log`).

**Seeds 43 / 44** (optional multi-seed; `--seed` overrides YAML):

```bash
cd "/Users/zubairishaq/Desktop/dr- research"
pkill -f 'phase_o|phase_p|src.train|src.evaluate|src.train_lodo|build_preprocess' 2>/dev/null; sleep 2

nohup caffeinate -is bash scripts/keepalive_phase_p_seeds.sh 43 44 \
  >> logs/keepalive_phase_p_seeds_outer.log 2>&1 &
echo $! > logs/keepalive_phase_p_seeds.pid

# watch
tail -f logs/keepalive_phase_p_seeds.log
# per seed:
tail -f logs/phase_p_lodo_s43.log
# done when: KEEPALIVE_P_SEEDS_ALL_OK
# per-seed: experiments/results/phase_p_lodo_s{43,44}.md
# mean±std: experiments/results/phase_p_lodo_seed_summary.md
```

Single seed (legacy): `bash scripts/keepalive_phase_p.sh 42`

## Stop
```bash
pkill -f 'keepalive_phase_o|keepalive_phase_p|run_phase_o|run_phase_p|src.train_lodo|src.train|src.evaluate'
```

## Notes
- Both use CPU · bs=4 · freeze_bn · chunked resume (jetsam-safe).
- LODO **breaks** the main-paper rule “never train on externals” — report as optional ablation.
- Do **not** run O and P at the same time on 8GB RAM.
- Seeds 43/44 need H2 externals for the same seed in the summary table (`hybrid_h2_s{43,44}_v1`); LODO runs themselves do not depend on H2.
