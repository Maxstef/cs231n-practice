# PyTorch gradient methods and properties

Autograd records operations so it can calculate derivatives of a selected
**scalar** score or loss. A tensor created directly with
`requires_grad=True` is usually a **leaf**. A result computed from it is an
**intermediate (non-leaf)** tensor.

| Item | Meaning in our notebooks |
| --- | --- |
| `x.requires_grad` | Whether operations on `x` participate in autograd |
| `x.is_leaf` | Whether `x` is a graph leaf; trainable parameters and a fresh saliency input are leaves |
| `x.grad_fn` | The operation that created a non-leaf tensor; usually `None` for a leaf |
| `x.grad` | Gradient stored **on** the tensor after backpropagation; usually populated for gradient-requiring leaves |
| `score.backward()` | Computes gradients and **accumulates** them into leaf `.grad` fields; returns `None` |
| `x.retain_grad()` | Also store `.grad` for a non-leaf intermediate tensor, such as a CNN feature map |
| `model.zero_grad(set_to_none=True)` | Clear the model **parameters'** accumulated gradients before another backward pass; does not clear an input tensor's `.grad` |

For input saliency, make a fresh leaf input and inspect its gradient:

```python
input_for_grad = image.detach().clone().requires_grad_(True)
model.zero_grad(set_to_none=True)
score = model(input_for_grad)[0, class_id]  # one scalar
score.backward()                           # returns None
pixel_gradients = input_for_grad.grad       # same shape as input
```

For Grad-CAM, the feature map is computed by the network, so it is **non-leaf**:

```python
features = model.features(input_for_grad)
features.retain_grad()  # request storage of this intermediate gradient
score = model.classifier(model.pool(features).flatten(1))[0, class_id]
score.backward()
feature_gradients = features.grad  # same shape as features
```

An alternative is `torch.autograd.grad(score, features)`: it **returns** the
requested derivative directly instead of accumulating it into `.grad` fields.
Our reusable attribution functions use this approach, so they do not need to
reset parameter gradients. `backward()` is convenient for training; `autograd.grad()`
is convenient when we only want selected derivatives.

Use `with torch.no_grad():` for an inference-only forward pass, such as the
many occluded-image predictions in occlusion sensitivity. Do **not** put the
saliency or Grad-CAM forward pass inside it: those methods need an autograd
graph. Also, `model.eval()` changes dropout/batch-normalization behavior; it
does **not** turn gradient tracking off.

Official references: [PyTorch leaf/non-leaf tutorial](https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html),
[`Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html),
[`Tensor.retain_grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html),
[`Module.zero_grad`](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html),
[`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html),
and [`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).
