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
