# Activation functions

An activation function applies a nonlinear transformation between learned affine
layers. Without it, stacking layers does not make the model more expressive:

$$
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2),
$$

which is still one affine transformation.

A nonlinearity lets a network build nonlinear decision boundaries. Geometrically,
the hidden layer can transform the input representation so classes that were not
linearly separable become easier for the final linear layer to separate.

## ReLU

The rectified linear unit is

$$
\mathrm{ReLU}(x)=\max(0,x).
$$

Its derivative is

$$
\mathrm{ReLU}'(x)=
\begin{cases}
1, & x>0,\\
0, & x<0.
\end{cases}
$$

It is not differentiable exactly at zero. Implementations conventionally choose a
subgradient, commonly zero; the exact-zero case rarely changes ordinary training.

ReLU is popular because it is simple, inexpensive, and does not shrink positive
gradients. It also makes the network piecewise linear: within a fixed pattern of
active ReLU units, the network behaves like an affine function, but different
patterns create different linear regions.

### Dead ReLUs

If a unit's pre-activation is negative for every training example, its output and
local derivative are both zero. It then receives no gradient through ReLU and may
remain inactive. This can be encouraged by an excessively large learning rate or
a strongly negative bias. A negative pre-activation for one example does **not**
stop the entire model; it only blocks that unit's gradient on that example.

## Leaky ReLU

Leaky ReLU retains a small slope $\alpha$ on the negative side:

$$
\mathrm{LeakyReLU}(x)=\max(\alpha x,x),
\qquad 0<\alpha\ll1.
$$

Its negative-side gradient is $\alpha$ rather than zero, reducing the chance that
a unit becomes permanently inactive. The slope is another design choice.

## Sigmoid

The sigmoid function is

$$
\sigma(x)=\frac{1}{1+e^{-x}},
$$

with derivative

$$
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

It maps values to $(0,1)$, which is useful when an output represents a binary
probability or an independent gate. As a hidden activation, it has disadvantages:
large positive or negative inputs saturate, their gradients become very small, and
the outputs are not zero-centered.

“Sigmoid” names the function. “Logistic regression” names a model that commonly
uses it to convert one score into a binary probability.

## Tanh

The hyperbolic tangent maps values to $(-1,1)$:

$$
\tanh'(x)=1-\tanh^2(x).
$$

Its outputs are zero-centered, but it still saturates for inputs with large
magnitude and can therefore produce small gradients.

## ELU

The exponential linear unit is

$$
\mathrm{ELU}(x)=
\begin{cases}
x, & x>0,\\
\alpha(e^x-1), & x\leq0.
\end{cases}
$$

It is smooth at zero when $\alpha=1$ and allows negative outputs. Its negative side
saturates, and computing an exponential is more expensive than ReLU.

## GELU

The Gaussian error linear unit is

$$
\mathrm{GELU}(x)=x\Phi(x),
$$

where $\Phi$ is the standard normal cumulative distribution function. It behaves
like a smooth, input-dependent gate rather than a hard threshold and is common in
Transformer architectures.

## SiLU or Swish

The sigmoid linear unit is

$$
\mathrm{SiLU}(x)=x\sigma(x).
$$

It is smooth, allows small negative outputs, and is used in several modern vision
architectures. Unlike ReLU, its derivative requires more computation.

## Hidden activations versus output transformations

Their roles should not be mixed up:

- ReLU and its alternatives usually transform hidden features.
- Sigmoid can turn one score into a binary probability or several independent
  scores into independent probabilities.
- Softmax turns class scores into probabilities that sum to one for a mutually
  exclusive multiclass problem.

There is no universally best activation. ReLU is a strong baseline for the small
network in this project; architecture, initialization, normalization, and the task
can make other choices preferable.

## Key takeaway

The activation function prevents a stack of affine layers from collapsing into one
affine layer. Its forward behavior determines the representations a network can
build, and its derivative determines how gradients flow backward.
