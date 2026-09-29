# Data parallelism, FSDP, and HSDP

## Data parallelism

In ordinary **data parallelism (DP)**, every GPU holds a complete copy of the
model and optimizer. A global batch is divided among $M$ workers:

```text
GPU 1: model copy + local minibatch -> local gradients
GPU 2: model copy + local minibatch -> local gradients
...
all GPUs: average gradients -> identical parameter update
```

If each worker processes $N$ examples, the effective global batch contains
$MN$ examples. Because differentiation and averaging are linear, averaging the
workers' gradients gives the gradient of the combined mean minibatch loss.

DP is simple and gives each worker independent forward and backward work. Its
main limitation is memory: every GPU stores all parameters, gradients, and
optimizer states. Adding GPUs increases batch capacity but does not allow a
model larger than one GPU's available model-state memory.

## Fully Sharded Data Parallelism

**FSDP** shards model state across workers. Each parameter shard, its gradient,
and its optimizer state have an owning GPU. For a layer:

1. workers gather the required parameter shards;
2. every worker computes that layer for its local minibatch;
3. temporary full parameters are released;
4. backward computation produces local gradients;
5. gradients are reduced and returned to the shard owners;
6. each owner updates its local shard.

Thus FSDP preserves data-parallel computation while avoiding a persistent full
copy of model state on every GPU. The price is more communication and more
complex scheduling. Prefetching the next layer's weights and overlapping
communication with computation are important for performance.

## Hybrid Sharded Data Parallelism

Sharding across an extremely large group can make every layer communicate over
slow links. **HSDP** divides the cluster into groups:

- use FSDP within each well-connected group;
- replicate the sharded model across groups;
- use data-parallel synchronization between groups.

This trades some extra replication for less expensive communication. It also
illustrates a general principle: different forms of parallelism can be composed
along different cluster dimensions.

## Comparison

| Method | Model state per GPU | Main communication | Best fit |
| --- | --- | --- | --- |
| DP | Full copy | Gradient synchronization | Model fits on one GPU |
| FSDP | Sharded | Parameter gathering and gradient reduction | Model state is too large for one GPU |
| HSDP | Sharded within replicated groups | Fast intra-group sharding plus inter-group synchronization | Very large hierarchical clusters |

## Source

- Stanford CS231n Spring 2025, Lecture 11: Large Scale Distributed Training,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
