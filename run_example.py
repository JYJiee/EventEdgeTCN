"""Run in PyCharm without arguments; edit the settings below for your own input."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eventedgetcn import (
    EdgeEventTCN, ModelConfig, decode_event_state,
    extract_aux_batch, find_activation_segments,
)

ROOT = Path(__file__).resolve().parent
INPUT_CSV = None  # None: synthetic demonstration; otherwise a path to your CSV.
CSV_COLUMN = "aggregate_w"
OUTPUT_DIR = ROOT / "outputs"
BATCH_SIZE = 32
SYNTHETIC_ENDPOINTS = 180
# For your own home, set these from its training/onboarding period, not TEST.
NORMALIZATION_MEAN_W = 359.4230594634287
NORMALIZATION_STD_W = 425.43300148293764


def load_model():
    config = json.loads((ROOT / "models/fridge/model_config.json").read_text())
    checkpoint = ROOT / "models/fridge/target_finetuned.pt"
    expected_hash = config["checkpoint_sha256"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Fridge checkpoint hash mismatch.")
    model = EdgeEventTCN(ModelConfig(**config["model"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    decoder = json.loads((ROOT / "models/fridge/decoder_config.json").read_text())["calibrated"]
    return model, decoder


def make_synthetic_aggregate(ws, endpoints=SYNTHETIC_ENDPOINTS):
    rng = np.random.default_rng(42)
    t = np.arange(ws + endpoints)
    background = 190.0 + 20.0 * np.sin(t / 25.0)
    cycling_load = 110.0 * ((t % 75) < 28)
    other_load = 800.0 * ((t >= ws + 60) & (t < ws + 70))
    return (background + cycling_load + other_load + rng.normal(0, 4, len(t))).astype(np.float32)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or CSV_COLUMN not in reader.fieldnames:
            raise ValueError(f"CSV must contain a {CSV_COLUMN!r} column.")
        values = []
        for row in reader:
            value = row[CSV_COLUMN]
            values.append(float(value) if value and value.strip() else np.nan)
    return np.asarray(values, dtype=np.float32)


def predict(aggregate, model, decoder, mean_w, std_w, batch_size=BATCH_SIZE):
    aggregate = np.asarray(aggregate, dtype=np.float32)
    ws = model.cfg.ws
    if aggregate.ndim != 1 or len(aggregate) <= ws:
        raise ValueError(f"Provide a one-dimensional series with more than {ws} rows.")
    if not np.isfinite(mean_w) or not np.isfinite(std_w) or std_w <= 0:
        raise ValueError("Normalization mean must be finite and std must be positive.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    valid = np.isfinite(aggregate)
    # Never remove missing rows: their positions preserve the dense 60 s timeline.
    safe = np.where(valid, aggregate, mean_w).astype(np.float32)
    normalized = ((safe - mean_w) / std_w).astype(np.float32)
    all_windows = np.lib.stride_tricks.sliding_window_view(normalized, ws)[:-1]
    history_valid = np.lib.stride_tricks.sliding_window_view(valid, ws)[:-1].all(axis=1)
    endpoints = np.arange(ws, len(aggregate), dtype=np.int64)[history_valid]
    if not len(endpoints):
        raise ValueError("No complete, finite exclusive-past windows are available.")
    aux = extract_aux_batch(normalized, ws)[history_valid]
    windows = all_windows[history_valid]
    probabilities, powers = [], []
    with torch.inference_mode():
        for start in range(0, len(endpoints), batch_size):
            stop = start + batch_size
            seq = torch.from_numpy(windows[start:stop].copy()).unsqueeze(1)
            features = torch.from_numpy(aux[start:stop].copy())
            logits, power = model(seq, features)
            probabilities.append(torch.sigmoid(logits).numpy())
            powers.append(power.numpy() * model.cfg.power_scale)
    probability = np.concatenate(probabilities)
    raw_power = np.concatenate(powers).clip(min=0)
    state = decode_event_state(probability, time_index=endpoints, **decoder)
    return {
        "endpoint": endpoints,
        "probability": probability,
        "decoded_state": state,
        "raw_power_w": raw_power,
        "gated_power_w": raw_power * state,
    }


def main():
    torch.set_num_threads(1)
    model, decoder = load_model()
    aggregate = (
        make_synthetic_aggregate(model.cfg.ws)
        if INPUT_CSV is None else load_csv(INPUT_CSV)
    )
    prediction = predict(
        aggregate, model, decoder, NORMALIZATION_MEAN_W, NORMALIZATION_STD_W
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "prediction_endpoint", "latest_input_endpoint", "on_probability",
            "decoded_state", "raw_power_w", "gated_power_w",
        ])
        for i, endpoint in enumerate(prediction["endpoint"]):
            writer.writerow([
                int(endpoint), int(endpoint - 1),
                float(prediction["probability"][i]),
                int(prediction["decoded_state"][i]),
                float(prediction["raw_power_w"][i]),
                float(prediction["gated_power_w"][i]),
            ])
    segments = find_activation_segments(
        prediction["decoded_state"], time_index=prediction["endpoint"]
    )
    events = [
        {"start_endpoint": int(prediction["endpoint"][start]),
         "end_endpoint_inclusive": int(prediction["endpoint"][end])}
        for start, end in segments
    ]
    summary = {
        "input": "synthetic" if INPUT_CSV is None else "user_csv",
        "sampling_seconds": 60,
        "input_contract": "x[t-W:t] predicts y[t]; latest input is x[t-1]",
        "normalization_mean_w": NORMALIZATION_MEAN_W,
        "normalization_std_w": NORMALIZATION_STD_W,
        "predicted_endpoints": len(prediction["endpoint"]),
        "events": events,
    }
    (OUTPUT_DIR / "events.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Input: {summary['input']}; predicted endpoints: {len(prediction['endpoint'])}")
    print(f"Decoded events: {len(events)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {OUTPUT_DIR / 'events.json'}")


if __name__ == "__main__":
    main()

