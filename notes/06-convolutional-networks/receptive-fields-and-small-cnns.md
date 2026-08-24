# Receptive fields and small CNNs

A unit's receptive field is the region of the original input that can influence
that unit. Although one convolution looks only at a small patch, stacking local
operations lets deeper features combine information from progressively larger
regions.

## Receptive-field size and jump

Track two quantities at layer $l$:

- $R_l$: receptive-field width measured in original input pixels;
- $J_l$: jump between adjacent features, also measured in original input pixels.

Starting with

$$
R_0=1,
\qquad
J_0=1,
$$

a layer with kernel size $K_l$ and stride $S_l$ gives

$$
R_l=R_{l-1}+(K_l-1)J_{l-1},
$$

$$
J_l=J_{l-1}S_l.
$$

Receptive field describes **how much** of the original image one feature sees. Jump
describes **how far that receptive field moves** when moving by one position in the
current feature map.

For stride-one convolutions with equal kernel size $K$, jump remains one. After $L$
layers,

$$
R_L=1+L(K-1).
$$

Three $3\times3$ convolutions therefore produce receptive fields of $3\times3$,
$5\times5$, and $7\times7$.

## Example containing pooling

Consider:

```text
3x3 convolution, stride 1
2x2 pooling, stride 2
3x3 convolution, stride 1
```

The history is

```text
input                         (R, J) = (1, 1)
after convolution             (R, J) = (3, 1)
after pooling                 (R, J) = (4, 2)
after convolution             (R, J) = (8, 2)
```

The final tuple `(8, 2)` means an $8\times8$ receptive field whose neighboring
positions are shifted by two original pixels. It does not mean an $8\times2$
field.

## Why stack small filters?

Several small convolutions can cover the same region as one larger convolution
while adding nonlinearities between operations. They also often require fewer
parameters.

For equal input and output channel count $C$, one $5\times5$ convolution uses
weights proportional to

$$
25C^2,
$$

while two $3\times3$ convolutions use

$$
18C^2.
$$

The two-layer version also contains two activation functions, allowing a more
expressive transformation.

## A small CNN shape flow

The project baseline is

```text
convolution → ReLU → max pooling
→ flatten
→ affine → ReLU
→ affine → softmax loss
```

For input $(N,C,H,W)$ and $F$ filters, convolution creates spatial feature maps
rather than immediately flattening pixels. Pooling reduces their resolution. If
the pooled tensor has shape

$$
(N,F,H_{pool},W_{pool}),
$$

then the first affine layer receives

$$
D_{flat}=F H_{pool}W_{pool}
$$

features per example. Only the last three axes are flattened; the batch axis must
remain separate.

## Why activations remain necessary

A sequence of convolutional operations without nonlinear activations is still a
linear transformation. ReLU between convolutions prevents the stack from
collapsing into one linear operation and lets progressively deeper features model
more complicated patterns.

## Verifying an assembled CNN

Layer tests do not guarantee correct composition. A complete CNN should also pass:

1. shape checks at every stage;
2. end-to-end numerical gradient checks for every parameter;
3. a small-dataset overfitting test;
4. inspection of loss curves, predictions, and early feature maps.

Successful overfitting shows that the implementation, gradients, optimizer, and
capacity can fit the selected examples. It does not demonstrate generalization to
unseen images.

## Connection to this project

- Notebook 16 introduces receptive fields and jumps.
- Notebook 20 assembles and gradient-checks the complete baseline CNN.
- `cs231n_practice/classifiers/cnn.py` contains the reusable `SmallConvNet`.

## Key takeaway

CNNs build spatial hierarchies: early units see small local regions, while deeper
units combine them into features with broader context. Careful shape tracking links
those spatial representations to the final classifier.
