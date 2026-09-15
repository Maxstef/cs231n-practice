# PyTorch: detach, cpu, and clone

A PyTorch tensor carries more than values: it may live on a GPU and be connected
to an **autograd graph** used for gradients. A NumPy-style `tensor.copy()` is not
a PyTorch Tensor method. PyTorch separates three decisions:

| Method | What it changes | What it does **not** guarantee |
| --- | --- | --- |
| `x.detach()` | Disconnects the result from the autograd graph | Does **not** copy the values; storage is shared |
| `x.clone()` | Makes separate tensor storage | Does **not** disconnect gradients; cloning is differentiable |
| `x.cpu()` | Moves the tensor to CPU if needed | Does **not** detach it or guarantee a fresh copy if already on CPU |

For a stable snapshot of an intermediate activation in a forward hook:

```python
snapshot = activation.detach().clone()
```

`detach()` says “do not backpropagate through this snapshot”; `clone()` says
“later changes to the original storage should not change my snapshot.” The
order is intentional: `activation.clone()` alone would still track gradients.

For input-gradient saliency, we need a **new independent input** that *does*
receive gradients:

```python
saliency_input = image.detach().clone().requires_grad_(True)
```

This starts a new gradient path at `saliency_input` without modifying the
original image tensor. To display a result or convert it to NumPy:

```python
display_array = result.detach().cpu().numpy()
```

`detach()` removes the gradient connection; `cpu()` ensures NumPy can access
CPU memory; `numpy()` changes the container type. If you need an independent
NumPy array rather than a possible shared-memory view, add `.copy()` **after**
`.numpy()`.

In short: **detach = gradient relationship; clone = storage independence;
cpu = device location.** Use only the steps your task needs.

Official references: [PyTorch `detach`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html),
[PyTorch `clone`](https://docs.pytorch.org/docs/stable/generated/torch.clone.html),
[PyTorch `cpu`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html),
and [PyTorch `numpy`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html).
