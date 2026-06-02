# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **monorepo for an A-share (Chinese stock market) machine-learning quant trading system**, split into three cooperating subprojects. The codebase and all docs are **Simplified Chinese** — keep comments, docstrings, and user-facing strings in Chinese. The environment is **Windows** (PowerShell; `.cmd` launchers throughout), Python.

```
stock/
├── qlib_quant/   # CORE: Qlib-based ML pipeline (data → train → predict → backtest → push)
├── qlib_ui/      # Streamlit web UI for browsing strategies, configs, and backtest results
├── qmt/          # QMT (迅投) data downloader + live position monitor (real broker terminal)
└── start-claude.cmd   # launches `claude --dangerously-skip-permissions`
```

There is **no root-level build/test/package** — each subproject is run independently from its own directory. Work inside the relevant subproject.

## The three subprojects and how they connect

The data flow ties them together; understanding it is the key to the repo:

```
qmt/  ──downloads A-share data──►  qlib_quant/data/qlib_data/cn_data/  (shared Qlib binary store)
                                          │
                          qlib_quant/  trains models, predicts next-day winners,
                                       backtests, and writes prediction/selection outputs
                                          │
                ┌─────────────────────────┴─────────────────────────┐
                ▼                                                     ▼
   qlib_ui/  reads qlib_quant configs &              qmt/strategy_monitor.py  loads qlib_quant's
   backtest results to visualize them                selection output and monitors live positions
```

- **`qlib_quant` is the center of gravity.** The other two read from or write to its `data/` and outputs. When in doubt, start there.
- **`qmt` writes into `../qlib_quant/data/qlib_data/cn_data`** (see `qmt/qmt_data_downloader.py --qlib-dir`). It requires the QMT broker client running with `xtquant`; without it, scripts fall back to mock data.
- **`qlib_ui` expects `qlib_quant` to be its sibling directory** (`../qlib_quant`) and reads its strategy YAMLs and `backtest_results/`.

## Per-subproject documentation (read these first)

Each subproject has its own authoritative docs — prefer them over re-reading code:

- **`qlib_quant/CLAUDE.md`** — detailed guidance for the core system (Chinese). It directs you to **`qlib_quant/docs/INDEX.md`**, the documentation hub (architecture, methodology, guides, status). For any qlib_quant work, the project's own rule is: **check docs first, then ask, then read code.**
- `qlib_quant/README.md`, `qlib_quant/QUICKSTART.md`, `qlib_quant/TROUBLESHOOTING.md`
- `qlib_ui/README.md`, `qlib_ui/QUICKSTART.md`, `qlib_ui/EXAMPLES.md`
- `qmt/README.md`, `qmt/TROUBLESHOOTING.md`

## Common commands

Each subproject installs its own deps (`pip install -r requirements.txt`) and runs from its own directory.

### qlib_quant (core ML pipeline)
```bash
cd qlib_quant
python scripts/daily_update.py                              # daily data update (AKShare → Qlib bin)
python runners/train.py    --config config/models/lgb_10d.yaml   # train
python runners/predict.py  --config config/models/lgb_10d.yaml   # next-day stock selection
python runners/workflow.py --config config/models/lgb_10d.yaml   # train + predict + backtest
python runners/backtest.py --config config/models/lgb_10d.yaml   # backtest
python walkforward/run_walkforward.py --config config/models/lgb_10d.yaml  # walk-forward validation
python push/daily_push.py                                   # generate & push daily picks
```
Model behavior is driven by **`config/models/*.yaml`** (dataset windows, Alpha158/Alpha360 features, LightGBM/CatBoost params, label). Global paths/trading params live in **`config/config.py`** (`PROJECT_ROOT`, `QLIB_DATA_PATH = data/qlib_data/cn_data`).

### qlib_ui (Streamlit visualization)
```bash
cd qlib_ui
streamlit run app.py          # or: python run.py  (serves http://localhost:8501)
```

### qmt (data download / live monitor) — requires QMT client running
```bash
cd qmt
python qmt_data_downloader.py --incremental                # incremental data pull
python qmt_data_downloader.py --convert-only               # convert raw → Qlib format
python strategy_monitor.py                                 # live position monitor + desktop alerts
```

## Working notes

- **Run scripts from the subproject root**, not from `stock/`. Paths in `config/config.py` and the cross-project links (`../qlib_quant/...`) assume this.
- **`config/models/` holds a very large number of strategy YAMLs** (`qvf_*`, `push25_*`, `cq*`, `hybrid*`, etc.) — they are variants/experiments. Don't enumerate them; pick the one named in the task or referenced by a runner, and read it directly.
- **No-look-ahead is a hard rule** for this domain: never let future data leak into features/labels. Respect A-share market mechanics — T+1 settlement and 10% / 20% (ChiNext, STAR) price limits — when touching trading or backtest logic.
- **Reproducibility**: `config/config.py` pins seeds (`RANDOM_SEED`, `PYTHONHASHSEED`). Preserve seeding when modifying training.
