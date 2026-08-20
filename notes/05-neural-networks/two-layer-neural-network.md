# Two-layer neural networks

A two-layer classifier extends a linear classifier by learning a hidden
representation before producing class scores. Here, “two-layer” counts the two
learned affine transformations; ReLU and softmax do not have learned weights.

## Architecture

For a batch $X$, the network used in this project is

$$
Z_1=XW_1+b_1,
$$

$$
H=\operatorname{ReLU}(Z_1),
$$

$$
S=HW_2+b_2.
$$

The shapes are

$$
X:(N,D), \quad W_1:(D,H), \quad b_1:(H),
$$

$$
W_2:(H,C), \quad b_2:(C), \quad S:(N,C).
$$

Here $N$ is the batch size, $D$ the input dimension, $H$ the hidden dimension, and
$C$ the number of classes.

The final scores are passed to softmax cross-entropy. Softmax is part of the loss
calculation during training; it is not another learned hidden layer.

## Why the hidden layer matters

A linear classifier learns one affine map from pixels to class scores. The first
layer of a neural network instead learns features, ReLU transforms them
nonlinearly, and the second layer combines those features into class scores.

The hidden dimension controls representation capacity. Too few units may be unable
to represent the pattern; more units increase capacity as well as computation and
the opportunity to overfit.

## Loss

A typical objective is

$$
L=L_{\text{data}}+
\lambda\left(\lVert W_1\rVert_2^2+\lVert W_2\rVert_2^2\right).
$$

Biases are commonly left unregularized. If the implementation places a factor of
$1/2$ in front of the regularization loss, its derivative is $\lambda W$ instead
of $2\lambda W$. Either convention is valid when loss and gradient agree.

## Backward pass

Backpropagation reverses the forward sequence:

1. Softmax cross-entropy produces `dS`.
2. The second affine layer produces `dW2`, `db2`, and `dH`.
3. ReLU masks `dH` wherever `Z1` was not positive, producing `dZ1`.
4. The first affine layer produces `dW1`, `db1`, and optionally `dX`.
5. Regularization gradients are added to `dW1` and `dW2`.

For the second layer,

$$
dW_2=H^T dS,
\qquad
db_2=\sum_n dS_n,
\qquad
dH=dS W_2^T.
$$

After ReLU,

$$
dZ_1=dH\odot\mathbf{1}[Z_1>0],
$$

and then

$$
dW_1=X^T dZ_1,
\qquad
db_1=\sum_n dZ_{1,n}.
$$

## A reliable implementation workflow

When building a small neural network from scratch:

1. Check the shape of every intermediate value.
2. Test each layer's forward and backward functions independently.
3. Use numerical gradient checking on a tiny double-precision problem.
4. Confirm that the model can overfit a very small training subset.
5. Tune hyperparameters using validation data.
6. Evaluate on the test set only after choices are finalized.

The tiny-subset test is especially useful. If a model with enough capacity cannot
drive training loss down on a handful of examples, the implementation or
optimization setup probably has a problem.

## CIFAR-10 perspective

The project network uses flattened raw pixels. It can learn nonlinear class
boundaries, but it discards the explicit spatial arrangement that convolutional
networks exploit. Its purpose is not state-of-the-art CIFAR-10 accuracy; it is a
transparent setting for understanding forward passes, backpropagation,
optimization, validation, and error analysis.

## Connection to this project

- Notebook 13 develops reusable affine and ReLU layers.
- Notebook 14 combines them into a two-layer network.
- Notebook 15 trains and evaluates that network on CIFAR-10.
- `cs231n_practice/neural_net.py` contains the reusable model implementation.

## Key takeaway

A two-layer network is a small composition of familiar operations. Its additional
power comes from the nonlinear hidden representation, while backpropagation makes
all parameters trainable using the same local-gradient rules used for simpler
computational graphs.
