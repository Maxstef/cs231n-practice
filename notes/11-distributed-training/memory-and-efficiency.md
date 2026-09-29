# Training memory and compute efficiency

Large-scale training is constrained by both memory capacity and execution
efficiency. A model may have enough total cluster memory yet still run slowly
because communication, small operations, or idle devices prevent the hardware
from reaching its peak throughput.

## What occupies training memory?

Important categories include:

- model parameters;
- parameter gradients;
- optimizer state, such as Adam's running first and second moments;
- activations saved for backward;
- temporary communication and operator buffers.

The exact byte count depends on numerical precision and implementation. Mixed
precision may also keep selected states in higher precision, so “parameter
count times parameter dtype” is not the total training memory.

## Activation checkpointing

Ordinary backpropagation saves intermediate activations from the forward pass
because backward formulas need them. Deep networks and long sequences can make
these activations larger than the persistent model state.

**Activation checkpointing** saves only selected boundary activations. During
backward, it reruns the forward computations between checkpoints to reconstruct
the missing values:

```text
more saved activations -> more memory, less recomputation
fewer saved activations -> less memory, more recomputation
```

This is a compute-for-memory trade-off. It does not change the mathematical
model or intended gradients, assuming recomputation reproduces the required
forward values.

## FLOPs and utilization

A **FLOP** is a floating-point operation. A multiply followed by an addition is
usually counted as two FLOPs. Device specifications report a theoretical peak,
but real workloads cannot sustain that rate continuously.

**Hardware FLOPs Utilization (HFU)** compares all executed floating-point work
with theoretical peak throughput. It may count recomputation and other executed
operations even when they do not represent the model's minimal forward and
backward work.

**Model FLOPs Utilization (MFU)** asks what fraction of peak throughput is used
for the model's useful, theoretical computation:

$$
\mathrm{MFU}
=\frac{\text{theoretical model FLOPs per iteration}}
{\text{peak FLOPs per second}\times\text{actual iteration time}}.
$$

MFU is lower when time is lost to communication, data loading, small inefficient
kernels, memory movement, synchronization, pipeline bubbles, or recomputation.

## Why larger hardware numbers do not guarantee proportional speedups

An operation can be **compute-bound** when arithmetic throughput is the main
limit, or **memory/communication-bound** when moving data is the main limit.
Accelerator compute capability can improve faster than memory or network
bandwidth, making efficient data movement increasingly important.

The practical goal is therefore not merely to use more GPUs. It is to choose a
parallelism layout, local batch size, precision, checkpointing policy, and data
pipeline that keep the GPUs doing useful work.

## Source

- Stanford CS231n Spring 2025, Lecture 11: Large Scale Distributed Training,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
