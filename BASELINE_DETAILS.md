# Baseline implementation details

This document describes the baseline implementations used in the paper's
matched REFIT comparison. It records their sources, implemented structures,
and adaptations to the prediction task. The runnable example in this repository
contains EventEdgeTCN only; the baseline descriptions accompany the paper.

## Shared task and evaluation settings

Each neural baseline uses a normalized aggregate window `x[t-W:t]` to predict
the state and power of one appliance in interval `t`. The window ends at `t-1`
and contains no aggregate observation from interval `t`. Sampling is at 60 s.

The learned methods use common household assignments, chronological data
allocations, state and power targets, and the joint training objective. Training
includes pretraining followed by supervised target-house adaptation. Checkpoint
selection uses the designated validation household; decoder calibration uses
target-house CAL after the network weights are frozen. The decoder family and
scoring masks are shared, with decoder parameters calibrated separately for
each method and appliance. Only EventEdgeTCN uses the 12 auxiliary statistics.
The paper's experimental-settings section gives the training schedule.

The comparison measures these implementations under the study's task and
training protocol. It does not reuse performance values published in the
baseline papers.

## EdgeTCNformer: adaptation of the authors' public implementation

**Source:** Moghadam and Fekr, EdgeTCNformer,
[DOI: 10.1016/j.enbuild.2026.117632](https://doi.org/10.1016/j.enbuild.2026.117632).
The model was adapted from the authors' public-code snapshot downloaded as
`EdgeTCNformer-main.zip`, specifically `model/edge_tcnformer.py`.

The adapted model retains the snapshot's sequence encoder:

- Two temporal convolutional blocks, each with three depthwise-separable
  convolution branches using kernel size 3 and dilations 1, 2, and 4. Branch
  outputs are averaged, followed by layer normalization and ReLU.
- A channel-attention module after each convolutional block, using global
  average pooling, a pointwise convolution, and sigmoid channel weights.
- A lightweight Transformer block with two attention heads, residual
  connections, layer normalization, and a two-layer feed-forward network.
- An embedding dimension of 64.

The public snapshot's decoder produces per-sample state and power sequences
for multiple appliances through pointwise convolutional heads. In this study,
that decoder is replaced by global average pooling over the encoded window,
dropout of 0.1, and two linear heads producing a state logit and a nonnegative
power estimate for one appliance. Sigmoid converts the state logit to an ON
probability. The power head uses the appliance's configured ReLU or softplus
activation. The model is trained with the study's joint objective and
target-house adaptation procedure. The encoder receives the exclusive-past
window defined above.

Thus, the principal architectural adaptation concerns pooling and the output
heads, while the encoder modules retain the downloaded implementation's
structure. The baseline is trained independently; the authors' trained
checkpoints are not used.

## DSTCN+NAM: independent implementation from the paper

**Source:** Li et al.,
[DOI: 10.1016/j.enbuild.2025.115700](https://doi.org/10.1016/j.enbuild.2025.115700).
This model was implemented from the paper's architectural description, rather
than adapted from the authors' executable training code.

The implementation uses four successive feature-extraction modules with
30 channels. Their convolution kernels are 10, 8, 6, and 5, and their paired
dilation rates are `(1, 8)`, `(2, 4)`, `(4, 2)`, and `(8, 1)`, respectively.
Each module sums two temporal branches and applies normalization-based
channel attention (NAM).

Each temporal branch contains two depthwise-separable convolutions, each
followed by single-group GroupNorm, ReLU6, and dropout of 0.1, with a pointwise
convolution on the residual path. Padding preserves the sequence length.
The NAM implementation uses batch normalization and normalized absolute
batch-normalization scale parameters to form sigmoid channel weights.

Global average pooling converts the final sequence representation to a
30-dimensional vector. Dropout of 0.1 and separate linear state and power
heads produce the scalar outputs required by the study. The state output
uses sigmoid, and the power output uses the appliance's configured nonnegative
activation. Training and decoder calibration follow the shared protocol.

These details specify the implementation evaluated here; the citation
identifies the source architecture rather than an execution of the authors'
original training pipeline.

## Other comparison methods

| Method and source | Implementation and task adaptation |
|---|---|
| Seq2Point ([Zhang et al., 2018](https://doi.org/10.1609/aaai.v32i1.11873)) | Independent convolutional adaptation: eight Conv1d layers with kernels 10, 8, 6, 5, 5, 5, 5, and 3; channels 48, 48, 48, 48, 48, 48, 24, and 12; GroupNorm and ReLU; global average pooling, dropout of 0.1, and scalar state/power heads. It replaces midpoint regression with the study's target alignment and uses Kaiming convolution initialization and Xavier linear initialization. |
| SGN ([Shin et al., 2019](https://doi.org/10.1609/aaai.v33i01.33011150)) | Independent implementation of the classification/regression gating structure. Separate convolutional branches use kernels 7, 5, 5, and 3 and channels 48, 48, 48, and 24, with ReLU, global average pooling, dropout of 0.1, and linear heads. The regression output is multiplied by the predicted ON probability, then processed by the study's event decoder for evaluation. |
| Window-GRU ([Krystalakos et al., 2018](https://doi.org/10.1145/3200947.3201011)) | Independent sliding-window recurrent implementation. A single-layer, unidirectional GRU with hidden size 48 maps its final hidden state through dropout of 0.1 to scalar state and power heads. |
| FHMM (Kolter and Johnson, 2011, REDD) | Independent two-chain factorial HMM representing the target appliance and the remaining aggregate background. Four joint states are filtered forward using `x[t-1]` to predict the target at `t`. Parameters are fitted on FT; the event decoder is calibrated on CAL. |
| Threshold | In-study reference using `x[t-1]`. A CAL-estimated background is subtracted from aggregate power; a calibrated residual threshold and transition temperature produce a soft state score. The decoded state gates the mean CAL appliance power during ON intervals for power evaluation. |

## Source snapshot identification

The EdgeTCNformer upstream commit identifier was not retained with the
downloaded archive. The following SHA-256 digests identify the source model
snapshot and the integrated baseline model definitions inspected for this
description; these files are not included in the inference example.

| File | SHA-256 |
|---|---|
| Downloaded `EdgeTCNformer-main/model/edge_tcnformer.py` | `2c73d398025ed2ea3c480f575e79da5db1ac79c30d89c649eeb84d1c94a3f807` |
| Integrated baseline `scripts/baselines/models.py` | `84d009082f4348638add6910d5301df966c8af556cb439dd96bac163c42d0cc7` |
