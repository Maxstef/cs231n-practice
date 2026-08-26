# From early CNNs to deeper architectures

CNN architecture evolved by repeatedly applying a few core ideas: local
connectivity, weight sharing, spatial downsampling, and increasingly deep stacks
of learned features.

## A short progression

### LeNet

LeNet established the familiar pattern of alternating convolution and
downsampling, followed by fully connected classification layers. It was built
for relatively small grayscale digit images.

### AlexNet

AlexNet showed that a much larger CNN trained with GPUs could perform extremely
well on ImageNet. Important ingredients included ReLU activations, data
augmentation, dropout, and overlapping max pooling.

### VGG

VGG made the architecture more regular: most feature extraction used repeated
$3 \times 3$ convolutions, with spatial resolution reduced periodically by
pooling. As height and width decreased, the number of channels generally
increased.

Depth and parameter count are related but are not the same thing. A deeper
network can sometimes contain fewer parameters than a shallower network if it
uses smaller kernels or a smaller classifier head.

## Why stack small kernels?

With stride 1 and no intervening downsampling, three $3 \times 3$ convolutions
have the same $7 \times 7$ receptive-field size as one $7 \times 7$
convolution:

$$
3 \longrightarrow 5 \longrightarrow 7.
$$

If every layer has $C$ input and output channels, the three small convolutions
use

$$
3(3 \cdot 3 \cdot C^2) = 27C^2
$$

weights, whereas one $7 \times 7$ convolution uses

$$
7 \cdot 7 \cdot C^2 = 49C^2.
$$

The stack also places nonlinear activations between the convolutions. It can
therefore represent a more complicated function than a single convolution,
while using fewer weights under this simplified equal-channel comparison.

The tradeoff is more sequential computation and potentially harder
optimization. Good initialization, normalization, and residual connections
help address that difficulty.

For the detailed receptive-field calculation, see
[Receptive fields and small CNNs](receptive-fields-and-small-cnns.md).

## A reusable architecture pattern

A typical image-classification CNN can be understood as three stages:

1. A **stem** produces the first learned feature maps.
2. Repeated **feature blocks** transform the representation and occasionally
   reduce its spatial resolution.
3. A **classification head** converts the final features into class scores.

Modern architectures mainly differ in their feature blocks, downsampling
strategy, and how the final spatial features are summarized.

## Further reading

- [Gradient-Based Learning Applied to Document Recognition](https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf)
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

