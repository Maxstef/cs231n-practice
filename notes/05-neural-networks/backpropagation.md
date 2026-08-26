# Backpropagation with vectors and matrices

Backpropagation is an efficient way to apply the chain rule through a computational
graph. More precisely, it is **reverse-mode automatic differentiation**: begin with
a scalar loss and propagate derivatives backward toward all inputs and parameters.

It is not an optimization algorithm. Backpropagation computes gradients; an
optimizer such as SGD uses those gradients to update parameters.

## Scalar loss, many parameters

Training usually produces one scalar loss $L$. A model may nevertheless contain
vectors, matrices, or higher-dimensional tensors. The gradient with respect to a
variable has the same shape as that variable:

- if $x$ is a scalar, $\partial L/\partial x$ is a scalar;
- if $x\in\mathbb{R}^D$, $\partial L/\partial x\in\mathbb{R}^D$;
- if $W\in\mathbb{R}^{D\times H}$, $\partial L/\partial W$ has shape $D\times H$.

Each entry answers: “If I slightly change this entry while keeping the others
fixed, how does the loss change?”

## What do `dout` and `dx` mean?

Suppose one layer computes

$$
out=f(x),
$$

and the complete network eventually produces scalar loss $L$. The layer's
backward function receives

$$
dout=\frac{\partial L}{\partial out}.
$$

`dout` is the **upstream gradient**: it tells the layer how sensitive the loss
is to every entry of that layer's output. Consequently, it always has the same
shape as `out`:

```text
dout.shape == out.shape
```

The layer combines `dout` with its local derivative and returns

$$
dx=\frac{\partial L}{\partial x}.
$$

`dx` tells us how sensitive the loss is to every entry of this layer's input,
so it has the same shape as `x`:

```text
dx.shape == x.shape
```

Therefore, `dout` is **not necessarily the gradient with respect to the model's
original input**. It is the gradient with respect to the output of whichever
operation is currently performing its backward pass.

### The names are relative to a layer

Consider two adjacent layers:

```text
x -- Layer A --> hidden -- Layer B --> out
```

Backward propagation moves in the opposite direction:

```text
dout -- Layer B backward --> dhidden -- Layer A backward --> dx
```

The intermediate gradient

$$
dhidden=\frac{\partial L}{\partial hidden}
$$

has two names depending on which layer we are discussing:

- it is the `dx` returned by Layer B, because `hidden` is Layer B's input;
- it is the `dout` received by Layer A, because `hidden` is Layer A's output.

Thus, one layer's `dx` becomes the preceding layer's `dout`. The underlying
gradient has not changed; only its role relative to the current layer has.

### Shape example: an affine layer

For

$$
out=xW+b,
$$

with

```text
x:     (N, D)
W:     (D, M)
b:     (M,)
out:   (N, M)
dout:  (N, M)
```

the backward formulas produce

```text
dx = dout @ W.T       -> (N, D)
dW = x.T @ dout       -> (D, M)
db = dout.sum(axis=0) -> (M,)
```

Each returned gradient has the same shape as the forward value with respect to
which it was differentiated.

### Why does a backward function need `dout`?

The local derivative of $f$ only describes how `out` changes when `x` changes.
Training needs to know how the final loss changes. The chain rule combines both:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial out}
\frac{\partial out}{\partial x}.
$$

The first factor is the incoming `dout`; the layer computes the appropriate
product with its local derivative to obtain `dx`. For tensor-valued functions,
this is conceptually a Jacobian product even though implementations avoid
constructing the full Jacobian.

### Branches accumulate input gradients

For a residual block

$$
out=F(x)+x,
$$

the same `dout` enters both paths. The shortcut contributes `dout` directly,
while the learned branch contributes the gradient obtained by backpropagating
through $F$:

$$
dx=dx_{branch}+dout.
$$

The result still has the shape of `x`. Addition is required because the same
input affects the loss through two separate computational paths.

### Memory rule

```text
dout: gradient arriving at a layer, shaped like its output
dx:   gradient leaving the layer backward, shaped like its input
```

## Jacobians and reverse-mode products

Suppose a function maps $x\in\mathbb{R}^D$ to $z\in\mathbb{R}^M$. Its complete
local derivative is the Jacobian

$$
J_{m,d}=\frac{\partial z_m}{\partial x_d},
$$

which has shape $M\times D$. If the upstream gradient is

$$
g_z=\frac{\partial L}{\partial z}\in\mathbb{R}^M,
$$

then the input gradient is

$$
g_x=J^Tg_z.
$$

In practice, we almost never construct the full Jacobian. Layer-specific backward
formulas calculate this contraction directly, saving memory and computation. For
matrix or tensor inputs, the same idea applies conceptually after flattening the
entries, while the implementation keeps their natural shapes.

## Affine layer derivation

For a batch of examples, use the project convention

$$
S=XW+b,
$$

with shapes

$$
X:(N,D), \qquad W:(D,M), \qquad b:(M), \qquad S:(N,M).
$$

Let the upstream gradient be

$$
dS=\frac{\partial L}{\partial S},
$$

also with shape $(N,M)$. One score is

$$
S_{n,m}=\sum_{d=1}^{D}X_{n,d}W_{d,m}+b_m.
$$

### Gradient with respect to the input

Each $X_{n,d}$ affects all $M$ outputs for example $n$:

$$
\frac{\partial L}{\partial X_{n,d}}
=\sum_m dS_{n,m}W_{d,m}.
$$

In matrix form,

$$
dX=dS\,W^T,
$$

and the shapes confirm it:

$$
(N,M)(M,D)=(N,D).
$$

### Gradient with respect to the weights

Each $W_{d,m}$ is shared by all $N$ examples, so their contributions add:

$$
\frac{\partial L}{\partial W_{d,m}}
=\sum_n X_{n,d}dS_{n,m}.
$$

Thus

$$
dW=X^T dS,
$$

with shapes

$$
(D,N)(N,M)=(D,M).
$$

### Gradient with respect to the bias

The same $b_m$ is broadcast to every example. Since its local derivative is one,
we sum the upstream gradients over the batch:

$$
db_m=\sum_n dS_{n,m}.
$$

In NumPy this is `db = np.sum(dS, axis=0)`.

## Shared values and accumulated gradients

Two common sums in backpropagation have the same underlying reason:

- a parameter shared across a batch receives contributions from every example;
- a value used by multiple graph branches receives contributions from every path.

In both cases, the total derivative is the sum of all ways the value affects the
loss.

## Gradient checking

A numerical gradient estimates one coordinate with finite differences:

$$
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i+h)-L(\theta_i-h)}{2h}.
$$

It is slow but useful for testing a new analytical backward pass on a small input.
Use double precision, avoid nondifferentiable points when possible, and remember
that making $h$ extremely small eventually increases floating-point cancellation
error.

## Key takeaway

Backpropagation repeatedly combines an upstream gradient with a local derivative.
For vector and matrix operations, matrix formulas efficiently perform the required
Jacobian products without building enormous Jacobian arrays.
