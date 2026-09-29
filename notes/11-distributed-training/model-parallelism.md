# Context, pipeline, and tensor parallelism

For a model with $L$ layers operating on tensors shaped approximately as
`(batch, sequence, dimension)`, work can be divided along several different
axes.

| Strategy | Split axis | What is distributed |
| --- | --- | --- |
| Data parallelism | Batch | Different examples |
| Context parallelism | Sequence | Tokens from one long sequence |
| Pipeline parallelism | Layers | Consecutive model stages |
| Tensor parallelism | Hidden dimension | Parts of operations inside a layer |

## Context parallelism

**Context parallelism (CP)** partitions a long sequence across GPUs. Token-wise
operations such as normalization and MLP projections are relatively direct to
split. Self-attention is harder because a query may need keys and values stored
on other workers.

Possible designs circulate blocks of keys and values between workers, or split
attention heads among workers. The goal is to make sequences fit in aggregate
memory while controlling communication and preserving numerically correct
attention.

## Pipeline parallelism

**Pipeline parallelism (PP)** assigns different groups of layers to different
GPUs. Activations cross GPU boundaries during the forward pass, and activation
gradients return during backward.

With only one minibatch, most stages wait while another stage works. This idle
time is the **pipeline bubble**. Dividing a batch into microbatches lets several
examples occupy different stages simultaneously, reducing—but not completely
eliminating—the bubble. Stages also need balanced compute; the slowest stage
limits throughput.

## Tensor parallelism

**Tensor parallelism (TP)** splits a single operation, commonly a linear layer,
across GPUs. If a weight matrix is partitioned by columns, workers compute
different output-feature blocks. If it is partitioned by rows, workers compute
partial contributions that must be summed.

Alternating compatible column- and row-partitioned layers can postpone a gather
and reduce communication. Nevertheless, TP usually communicates within every
Transformer block, so it benefits strongly from fast local interconnects.

## Multidimensional parallelism

The largest training jobs combine these methods rather than selecting only one:

```text
global GPU count = DP size x CP size x PP size x TP size
```

Each worker has a rank along every dimension. The exact configuration balances
model fit, sequence length, global batch size, communication topology, and
accelerator utilization.

## Key takeaway

Each strategy solves a different bottleneck: DP increases example throughput,
CP handles long sequences, PP divides layer depth, and TP divides wide layer
operations. Every split saves or distributes some resource while introducing
communication or scheduling costs.

## Source

- Stanford CS231n Spring 2025, Lecture 11: Large Scale Distributed Training,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
