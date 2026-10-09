# Verification

Validated on Python 3.11.15 using synthetic inputs only.

- The standalone entry runs from another working directory.
- Four unittest checks pass: exclusive-past network input, missing-history
  filtering, missing-time decoder boundaries, and power gating.
- Extracted architecture, auxiliary features, and decoder match the source
  release exactly on synthetic inputs.
- Checkpoint and decoder JSON are byte-identical copies of the source release.
- REFIT House 20 normalization metadata hash matches the locked manifest.

No training, real TEST scoring, device flashing or external publication was
performed during package preparation. The generated ZIP excludes demonstration
outputs and Python caches.

