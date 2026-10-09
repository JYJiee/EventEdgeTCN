# EventEdgeTCN example release

A standalone inference example accompanying EventEdgeTCN. This initial release
contains the core network, one final REFIT fridge checkpoint, its frozen CAL
decoder, and selected configuration metadata for the nine evaluated appliances.

## Run

Python 3.10+ is required; validated with Python 3.11.15.

```text
python -m pip install -r requirements.txt
python run_example.py
```

Alternatively, open this directory in PyCharm and click Run on
`run_example.py`. No command-line arguments or REFIT preprocessing caches
are required. The default demonstration uses synthetic aggregate power and
runs on CPU. Outputs are written to `outputs/predictions.csv` and
`outputs/events.json`; synthetic outputs are not paper accuracy results.

For your own input, edit `INPUT_CSV` and the normalization constants at the
top of the script. CSV format:

```csv
aggregate_w
315.0
420.0
389.0
```

Supply more than 240 chronological rows of total active power in watts, one
row per completed 60 s interval. Keep a blank value for a missing interval
instead of removing the row. Windows crossing missing values are skipped.
Raw sensor data must first be aggregated into this format. Use normalization
statistics from your home's training/onboarding data; the supplied constants
come from the locked REFIT House 20 fridge preprocessing metadata.

## Inference contract

For endpoint `t`, the network uses only `x[t-240:t]`, ending at `t-1`.
It produces an ON probability and a nonnegative power estimate. The frozen
sequence decoder reconstructs events and gates the power estimate.
Sequence decoding operates after predictions have been collected; it is
not an immediate streaming event decision. Event coordinates are zero-based
60 s endpoints, with inclusive end indices.

## Contents

- `eventedgetcn.py`: network, 12 auxiliary statistics, event decoder.
- `run_example.py`: synthetic or user-CSV inference, CSV and JSON outputs.
- `models/fridge/`: checkpoint, model metadata, calibrated decoder.
- `configs/appliance_reference.json`: selected architecture, split,
  ground-truth and final decoder parameters for all nine appliances.
- `tests/test_example.py`: input alignment, missing-time decoding,
  checkpoint loading and inference checks.
- `PROVENANCE.json`: source and artifact hashes.

The configuration reference records selected settings, not the entire
decoder search grid. Only the fridge checkpoint is included.
Baseline experiments, training orchestration, other checkpoints, datasets,
preprocessing caches, scheduling experiments and ESP32 firmware are outside
this example. An expanded release is planned after paper acceptance.
The example does not reproduce the paper's complete experimental tables.

Run the lightweight checks with:

```text
python -m unittest discover -s tests
```

## License and Citation

Code and the included checkpoint are provided under the MIT license in
`LICENSE`. REFIT data are not distributed; obtain them from their original
provider and comply with the dataset license. The manuscript is not covered
by this software license. Citation details will be added with publication.

