# Weight initialization

Weight initialization chooses a network's weights **before** training begins.
The weights must break symmetry, but their scale matters too: activations and
gradients should remain numerically useful as they pass through many layers.

## Why zero weights do not work for hidden layers

If every neuron in a hidden layer starts with the same weights, every neuron
produces the same output and receives the same gradient. Gradient descent keeps
them identical, so the layer behaves as though it had only one useful neuron.

Random weights break this symmetry. Merely choosing random numbers is not
enough, however; their variance should suit the layer and its activation.

## What can go wrong with the scale?

Consider a linear pre-activation:

$$
z = Wx.
$$

If weights are too small, repeated multiplications can make activations and
gradients shrink toward zero. If they are too large, activations and gradients
can grow rapidly and become unstable. These are commonly called **vanishing**
and **exploding** activations or gradients.

A useful initialization tries to keep activation variance at roughly the same
scale from one layer to the next.

## Xavier initialization

For linear or approximately symmetric activations such as `tanh`, a common
fan-in form of Xavier (Glorot) initialization is

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{1}{D_{\text{in}}}\right).
$$

Equivalently,

$$
\mathrm{std}(W) = \sqrt{\frac{1}{D_{\text{in}}}}.
$$

Here $D_{\text{in}}$, also called `fan_in`, is the number of inputs to one
neuron. Some Xavier variants use both `fan_in` and `fan_out` to balance forward
activations and backward gradients.

## Kaiming initialization for ReLU

A ReLU sets negative inputs to zero, so under the usual simplifying assumptions
about half of the signal is removed. Kaiming (or He) initialization compensates
with a factor of two:

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{D_{\text{in}}}\right),
$$

and therefore

$$
\mathrm{std}(W) = \sqrt{\frac{2}{D_{\text{in}}}}.
$$

This does not force every layer to have exactly identical statistics. It is a
principled starting point that makes deep ReLU networks easier to optimize than
an arbitrary fixed standard deviation.

For a convolution with kernel size $K_H \times K_W$ and $C_{\text{in}}$ input
channels,

$$
D_{\text{in}} = C_{\text{in}}K_HK_W.
$$

The same rule therefore applies to fully connected and convolutional layers;
only the definition of `fan_in` changes.

## Practical summary

- Do not initialize every hidden-layer weight to zero.
- Xavier initialization is a common choice for linear or `tanh`-like layers.
- Kaiming initialization is a common choice for ReLU-like layers.
- Biases can usually start at zero because random weights already break the
  symmetry between neurons.
- Initialization, normalization, residual connections, and the learning rate
  address related but distinct optimization concerns.

## Further reading

- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)

