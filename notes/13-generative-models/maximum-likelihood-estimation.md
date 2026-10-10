# Maximum likelihood estimation

Maximum likelihood estimation (MLE) chooses model parameters that make the
observed training data as likely as possible.

## Data and model notation

Write a dataset of $N$ observations as

$$
x^{(1)},x^{(2)},\ldots,x^{(N)}.
$$

Here:

- $x$ is one observation, such as an image;
- the superscript $(i)$ is the example index, not an exponent;
- $N$ is the number of training examples.

A parameterized model is written as

$$
p_\theta(x),
$$

where $p$ is a probability distribution or density and $\theta$ contains all
trainable parameters, such as neural-network weights and biases.

The value

$$
p_\theta(x^{(i)})
$$

describes how likely example $i$ is under the current model. For continuous
data it is a density value rather than the probability of one exact point.

## From individual examples to dataset likelihood

Training examples are usually assumed to be independently drawn from the same
distribution. Independence lets us multiply their individual likelihoods:

$$
p_\theta(x^{(1)},\ldots,x^{(N)})
=\prod_{i=1}^{N}p_\theta(x^{(i)}).
$$

The product symbol means

$$
\prod_{i=1}^{N}p_\theta(x^{(i)})
=p_\theta(x^{(1)})p_\theta(x^{(2)})\cdots p_\theta(x^{(N)}).
$$

When viewed as a **likelihood**, the observed data is fixed and the expression
is considered as a function of the unknown parameters $\theta$.

## Choosing the best parameters

The maximum-likelihood estimate is

$$
\theta^*
=\arg\max_\theta
\prod_{i=1}^{N}p_\theta(x^{(i)}).
$$

The notation means:

- $\theta$ is one possible parameter setting;
- $\max$ would return the largest likelihood value;
- $\arg\max$ returns the parameter setting that produces that value;
- $\theta^*$ denotes the selected optimal parameters.

For example, if the greatest value of $L(\theta)$ is $0.4$ and occurs at
$\theta=0.7$, then

```text
max L(theta)    = 0.4
argmax L(theta) = 0.7
```

## The logarithm transformation

Multiplying many probabilities can produce a number too small for floating-point
arithmetic. We take the logarithm of the likelihood instead. Because logarithm
is strictly increasing, it preserves which parameter setting is largest:

$$
a>b \quad\Longrightarrow\quad \log a>\log b.
$$

The rule

$$
\log(ab)=\log a+\log b
$$

changes the product into a sum:

$$
\log\left(\prod_{i=1}^{N}p_\theta(x^{(i)})\right)
=\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

Therefore, maximizing likelihood is equivalent to maximizing log-likelihood:

$$
\theta^*
=\arg\max_\theta
\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

A probability between zero and one has a nonpositive logarithm. A higher
probability produces a larger, less negative log-probability:

```text
p = 0.90 -> log(p) is approximately -0.105
p = 0.10 -> log(p) is approximately -2.303
```

## Negative log-likelihood

Neural networks are normally trained by minimizing a loss with gradient
descent. Negating log-likelihood converts its maximization into minimization:

$$
\mathcal{L}_{NLL}
=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

The symbols mean:

- $\mathcal{L}_{NLL}$ is negative log-likelihood loss;
- the minus sign converts maximization to minimization;
- the sum combines the examples;
- $1/N$ averages over the dataset;
- $\log p_\theta(x^{(i)})$ is example $i$'s log-likelihood.

Dividing by positive constant $N$ changes the scale but not the optimal
parameters. Averaging makes losses comparable across dataset and minibatch
sizes.

Gradient descent then performs

$$
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}_{NLL},
$$

where $\eta$ is the learning rate and $\nabla_\theta\mathcal{L}_{NLL}$ is the
gradient with respect to all model parameters. Decreasing NLL is equivalent to
increasing log-likelihood.

## Bernoulli example

Suppose $x\in\{0,1\}$ and $\theta$ is the probability of observing one:

$$
p_\theta(x=1)=\theta,
\qquad
p_\theta(x=0)=1-\theta.
$$

For the dataset

$$
x^{(1)}=1,\qquad x^{(2)}=1,\qquad x^{(3)}=0,
$$

the likelihood is

$$
L(\theta)=\theta\cdot\theta\cdot(1-\theta)
=\theta^2(1-\theta).
$$

Its log-likelihood is

$$
\log L(\theta)=2\log\theta+\log(1-\theta).
$$

Differentiate and set the result to zero:

$$
\frac{d}{d\theta}\log L(\theta)
=\frac{2}{\theta}-\frac{1}{1-\theta}=0.
$$

Rearranging gives

$$
2(1-\theta)=\theta,
\qquad
\theta^*=\frac{2}{3}.
$$

This agrees with the observed proportion: two of the three examples are one.

## Equivalent objectives

The complete transformation is

$$
\arg\max_\theta
\prod_{i=1}^{N}p_\theta(x^{(i)})
$$

$$
=\arg\max_\theta
\sum_{i=1}^{N}\log p_\theta(x^{(i)})
$$

$$
=\arg\min_\theta
\left[-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x^{(i)})\right].
$$

They have the same optimal parameters: likelihood uses a product,
log-likelihood uses a numerically stable sum, and NLL gives the minimization
objective normally used in code.

## Source

- Stanford CS231n Spring 2025, Lecture 13: Generative Models 1, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).

