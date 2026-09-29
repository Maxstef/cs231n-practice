# GPU hardware and cluster communication

Neural-network training contains many independent numerical operations, most
notably matrix multiplications. GPUs accelerate this workload by running many
operations in parallel rather than optimizing for the fast sequential execution
of a small number of instructions.

## Compute and memory inside a GPU

A GPU contains many **streaming multiprocessors**. These include general
floating-point cores and specialized **Tensor Cores** for small matrix
multiply-accumulate operations. Lower-precision formats such as FP16 or BF16
allow Tensor Cores to provide much higher throughput than ordinary FP32
operations.

Peak arithmetic throughput alone is not enough. Data must move through a memory
hierarchy:

```text
registers and cache <-> GPU high-bandwidth memory <-> other GPUs
```

Registers and cache are small and fast. High-bandwidth memory is much larger
but slower. Communication with another GPU is slower again, especially when it
crosses server, rack, or cluster boundaries.

## Why cluster topology matters

Large clusters are hierarchical: GPUs are grouped into servers, racks, pods,
and the full cluster. Links between nearby GPUs usually have more bandwidth and
lower latency than links between distant GPUs.

Consequently, distributed training is not just “divide the work equally.” A
good layout places communication-heavy groups on the fastest links and tries to
overlap communication with useful computation.

For example, tensor parallelism communicates frequently and is usually kept
among tightly connected GPUs. Data-parallel replicas communicate less often
and can be distributed across slower cluster boundaries.

## GPU versus TPU

GPUs are widely used parallel processors with matrix-specific hardware. TPUs
are custom accelerators designed around tensor computation and are also
organized into large connected systems. The same central constraints apply to
both: compute throughput, accelerator memory, and communication bandwidth.

## Key takeaway

A cluster may look like one enormous computer, but it has a strongly
non-uniform memory and communication hierarchy. Distributed algorithms must be
designed around that hierarchy.

Hardware specifications in lecture examples are snapshots in time; the
conceptual relationships are more durable than individual device numbers.

## Source

- Stanford CS231n Spring 2025, Lecture 11: Large Scale Distributed Training,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
