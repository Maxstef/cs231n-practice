# Gradient descent and stochastic gradient descent

## The optimization objective

Training adjusts model parameters, such as weights $W$, to minimize an
objective. For $N$ training examples, a common objective is

$$
L(W) = \frac{1}{N}\sum_{i=1}^{N} L_i(W) + \lambda R(W),
$$

where $L_i$ is the data loss for example $i$ and $R$ is a regularization
penalty. Gradient-based methods repeatedly move the parameters in a direction
that reduces this objective:

$$
W_{t+1} = W_t - \alpha \nabla_W L(W_t).
$$

Here $\alpha$ is the **learning rate**: it controls the size of each update.

## Full-batch gradient descent

Full-batch gradient descent computes the gradient using all $N$ training
examples before making one update:

$$
g_t = \frac{1}{N}\sum_{i=1}^{N}\nabla_W L_i(W_t) +
\lambda \nabla_W R(W_t).
$$

This gives the exact gradient of the finite training objective. Its direction
is stable, but every update becomes expensive when the dataset is large.

## Stochastic and minibatch gradient descent

Strictly speaking, **stochastic gradient descent (SGD)** uses one randomly
chosen example per update. In deep learning, however, *SGD* usually means
**minibatch SGD**: estimate the gradient from a small batch $B_t$:

$$
g_t = \frac{1}{|B_t|}\sum_{i \in B_t}\nabla_W L_i(W_t) +
\lambda \nabla_W R(W_t).
$$

Then update with

$$
W_{t+1} = W_t - \alpha g_t.
$$

If batches are sampled appropriately, $g_t$ estimates the full-data gradient.
Different minibatches give different estimates, so individual updates and the
recorded loss are noisy. This is expected rather than evidence that the code is
wrong.

## Why minibatches are the practical compromise

- A minibatch is much cheaper than the entire dataset.
- Matrix operations process a batch efficiently on modern hardware.
- It is less noisy than using just one example.
- Frequent updates usually make faster progress per unit of computation than
  waiting for a full-dataset gradient.

Increasing the batch size generally produces a less noisy gradient estimate,
but requires more memory and more computation per update. It may also require
retuning the learning rate.

## Essential vocabulary

- **Batch size:** number of training examples used for one gradient estimate.
- **Iteration** or **step:** one parameter update.
- **Epoch:** enough iterations to process approximately the whole training set
  once. With $N$ examples and batch size $B$, this is roughly $N/B$ steps.

## Comparison to remember

- **Gradient descent:** exact full-training-set gradient; stable but expensive.
- **Single-example SGD:** cheapest and noisiest gradient estimate.
- **Minibatch SGD:** efficient compromise and the usual meaning of SGD in deep
  learning.

The choice of gradient estimate is separate from the update rule. Vanilla SGD,
SGD with momentum, RMSProp, and Adam can all operate on minibatch gradients.

## In this project

Minibatch sampling and SGD training are explored in
[`09_softmax_linear_classifier_sgd.ipynb`](../../notebooks/09_softmax_linear_classifier_sgd.ipynb)
and compared across losses in
[`10_svm_vs_softmax_training.ipynb`](../../notebooks/10_svm_vs_softmax_training.ipynb).
