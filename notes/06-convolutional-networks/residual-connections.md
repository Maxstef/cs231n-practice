# Residual connections

Residual connections provide a direct path around one or more learned layers.
They make deep networks easier to optimize and give gradients an additional,
shorter route during backpropagation.

## The degradation problem

Making a plain network deeper does not necessarily reduce its **training**
error. Experiments with early very deep CNNs showed that some deeper plain
networks trained worse than shallower versions.

This is called the **degradation problem**. It differs from ordinary
overfitting:

- Overfitting means training performance improves while validation performance
  becomes worse.
- Degradation means optimization itself is harder, so even training performance
  can become worse.

In principle, a deeper network should be able to reproduce a shallower one by
making its extra layers implement the identity function. In practice, a stack
of ordinary nonlinear layers may find that identity mapping difficult to learn.

## Learn a residual instead of the full mapping

Suppose the desired mapping is $H(x)$. A residual block represents it as

$$
H(x) = F(x) + x,
$$

where $F(x)$ is the transformation learned by the block and $x$ travels through
the shortcut connection. Equivalently,

$$
F(x) = H(x) - x.
$$

If the best mapping is close to the identity, the block only needs to learn
$F(x) \approx 0$. This is generally easier than asking several nonlinear layers
to reconstruct $x$ exactly.

## Forward and backward views

For the simple block

$$
y = F(x) + x,
$$

an upstream gradient $dY$ reaches $x$ through both branches:

$$
dX = dY\frac{\partial F}{\partial x} + dY
   = dY\left(\frac{\partial F}{\partial x} + I\right).
$$

The $dY$ term comes from the identity shortcut. Even if the learned branch has
small or troublesome derivatives, the shortcut provides a direct gradient
path. The two contributions are **added** because the same input affects the
output through both computational paths.

This is the same branching rule described in
[Computational graphs](../05-neural-networks/computational-graphs.md) and
[Backpropagation with vectors and matrices](../05-neural-networks/backpropagation.md).

## When shapes differ

The addition requires $F(x)$ and the shortcut to have the same shape. If a block
changes the spatial size or number of channels, the shortcut can use a learned
projection, commonly a $1 \times 1$ convolution:

$$
y = F(x) + W_sx.
$$

The projection aligns the shapes. It is unnecessary when input and output
already match; an identity shortcut is then simpler and adds no parameters.

## What residual connections do not mean

- They do not make every residual block an identity mapping; they make identity
  easy to represent when useful.
- They do not eliminate the need for nonlinearities, initialization,
  normalization, or optimization.
- They do not guarantee that an arbitrarily deep model will generalize better.

Their central benefit is a parameter-free information and gradient path that
makes useful modifications to an existing representation easier to learn.

## Further reading

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

