# GPU versus TPU

Both GPUs and TPUs accelerate tensor computation, but they make different
engineering trade-offs.

## Architecture and performance

**GPU — Graphics Processing Unit.** A GPU is a broadly programmable parallel
processor. Its streaming multiprocessors contain general arithmetic cores,
registers, caches, and specialized Tensor Cores. This mixture supports matrix
operations as well as indexing, control flow, rendering, simulation, and custom
kernels. Modern GPU software supports several numerical formats, including
FP32 and lower-precision formats used for mixed-precision training.

This flexibility has a cost: operands and intermediate values must repeatedly
move among registers, caches, and high-bandwidth memory. Performance therefore
depends on both arithmetic throughput and data movement. Tensor Cores reduce
this gap for supported matrix and convolution operations, but irregular work
may still run on more general execution units.

**TPU — Tensor Processing Unit.** A Cloud TPU is an application-specific
integrated circuit designed by Google for machine-learning workloads. Its main
compute units are matrix multiplication units built from large **systolic
arrays** of connected multiply-accumulate cells.

During a matrix multiplication, values flow through neighboring cells and
partial results are accumulated as they move. This reuses data inside the array
instead of reading and writing every intermediate product to accelerator memory.
The result can be high throughput and energy efficiency for large, regular
matrix operations. TPUs also contain vector and scalar units for operations
that do not map to the matrix units.

A systolic array is specialized rather than universally faster. Frequent
branching, unsupported custom operations, small irregular workloads, or high
precision requirements may use the hardware less effectively. Actual GPU–TPU
performance also depends on model shapes, batch size, compiler behavior,
communication, software maturity, availability, and cost.

## Ecosystem and flexibility

| GPU | TPU |
| --- | --- |
| Broadly programmable accelerator | ML-focused application-specific accelerator |
| NVIDIA GPUs use CUDA and its mature kernel and library ecosystem | Cloud TPUs are programmed through the XLA compiler stack |
| Common in personal computers, workstations, servers, and clouds | Large training TPUs are primarily provisioned through Google Cloud |
| Widely supported by PyTorch, TensorFlow, JAX, and non-ML tools | Supports JAX, PyTorch through PyTorch/XLA, and supported TensorFlow configurations |
| Often easier for custom kernels, irregular algorithms, and mixed workloads | Particularly suitable for large, regular, compiler-friendly tensor workloads |

PyTorch code can therefore run on TPUs; it is not restricted to GPUs. However,
PyTorch/XLA compiles graphs for the XLA device, so code written around frequent
host interaction or unsupported custom operations may require adaptation.

## Which should be chosen?

Consider a **GPU** when broad software compatibility, local or on-premises
access, graphics and rendering, custom kernels, irregular computation, or an
existing CUDA-based stack matters most.

Consider a **TPU** when training a large matrix-dominated model on Google Cloud,
especially when the workload maps cleanly to XLA and can use large effective
batches or a connected TPU slice. This is a workload-dependent choice, not a
general rule that every large language model should use a TPU.

The same central constraints ultimately apply to both: compute throughput,
accelerator memory, interconnect bandwidth, software support, and how well the
workload maps to the hardware.

## Official references

- [Google Cloud TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Google guidance on when to use TPUs](https://docs.cloud.google.com/tpu/docs/intro-to-tpu)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/master/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

## Course source

- Stanford CS231n Spring 2025, Lecture 11: Large Scale Distributed Training,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
