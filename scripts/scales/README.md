# Workload scaling presets

Four `.env` files that scale every benchmark's `workload_value`
roughly geometrically so the prediction model can learn the
per-service scaling curve on each hardware target.

| Scale | Multiplier vs `large` | Approx. total runtime |
| --- | --- | --- |
| `small`  | ~0.2x  | 10-15 min |
| `medium` | ~0.5x  | 25-35 min |
| `large`  | 1.0x   | 45-55 min (baseline) |
| `xlarge` | ~2.0x  | 90-120 min |

`large` reproduces the original `.env` baseline used for the
4 reference runs (2026-05-20 / 21). Other scales are derived
from it so curves intersect cleanly when joined.

## Use

Either source the file before invoking the orchestrator:

```bash
set -a; source scripts/scales/medium.env; set +a
scripts/orchestrator.sh
```

…or use the batch runner which iterates all four:

```bash
scripts/run_scaled_batch.sh
```

The orchestrator records `WORKLOAD_SCALE` into the run's
`run_metadata.json` so `build_ml_dataset.py` can join scale
back onto each row.

## Adding a new scale

Drop another `.env` file in this directory and add it to the
`SCALES` default in `scripts/run_scaled_batch.sh`. Keep the
ratio between values reasonable across scales (linear or
geometric) so the model can extrapolate.
