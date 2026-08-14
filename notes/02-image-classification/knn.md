# k-Nearest Neighbors

## Core idea

k-nearest neighbors (kNN) is an **instance-based** classifier. Training does not
learn a parameter matrix; it stores labeled examples. For a new query, the
classifier:

1. computes its distance to every stored example;
2. selects the `k` smallest distances;
3. predicts the most frequent label among those neighbors.

This makes training cheap and prediction expensive. With `Q` queries, `S`
stored examples, and `D` features, direct distance calculation requires work
proportional to `Q × S × D`.

## L1 and L2 distances

For feature vectors $x,z\in\mathbb{R}^D$, the L1 or Manhattan distance is

$$
d_1(x,z)=\sum_{p=1}^{D}|x_p-z_p|.
$$

The L2 or Euclidean distance is

$$
d_2(x,z)=\sqrt{\sum_{p=1}^{D}(x_p-z_p)^2}.
$$

L1 adds absolute coordinate differences. L2 squares them before summation, so
large individual differences have greater influence. The metrics can therefore
produce different neighbor rankings and decision boundaries.

In two dimensions, points at a fixed L1 distance from the origin form a diamond;
points at a fixed L2 distance form a circle. This geometry illustrates how the
choice of metric changes what "near" means.

## Squared L2 distance

The square root can be omitted when ranking neighbors because it is strictly
increasing for nonnegative inputs:

$$
\|x-z\|_2^2=\|x\|_2^2+\|z\|_2^2-2x^Tz.
$$

For query matrix $X_q\in\mathbb{R}^{Q\times D}$ and stored matrix
$X_s\in\mathbb{R}^{S\times D}$:

- query squared norms have shape `(Q, 1)`;
- stored squared norms have shape `(1, S)`;
- cross-products $X_qX_s^T$ have shape `(Q, S)`;
- broadcasting produces the complete `(Q, S)` distance matrix.

This avoids allocating a much larger `(Q, S, D)` difference array.

## The role of `k`

- Small `k` produces flexible, local decisions but is sensitive to individual
  noisy examples.
- Large `k` averages over more examples, producing smoother decisions but
  potentially ignoring useful local structure.
- `k` is a hyperparameter and must be chosen with validation data.

Vote ties require a convention. Our implementation selects the smallest class
ID. Class IDs have no semantic ordering, so this rule provides reproducibility,
not a statistically justified preference.

## Limitations for raw images

Raw-pixel distances are sensitive to translation, pose, illumination,
background, scale, and cropping. Two images of the same semantic class may be
far apart, while images from different classes may have similar colors and
layouts.

In our CIFAR-10 experiment, raw-pixel kNN achieved 26.4% accuracy on a
500-image test subset. It also strongly favored some predicted classes even
though the stored training subset was approximately class-balanced. This
suggests that raw-pixel geometry, not only class frequency, drove the bias.

## Related project material

- `notebooks/01_cifar10_knn_distances.ipynb`
- `notebooks/02_knn_vectorized_distances.ipynb`
- `notebooks/03_knn_prediction_and_validation.ipynb`
- `cs231n_practice/classifiers/knn.py`
- `tests/test_knn.py`

## Review questions

1. Why is kNN described as instance-based and non-parametric?
2. Why can squared L2 distance replace L2 distance during neighbor ranking?
3. How do L1 and L2 differ in their response to one large feature difference?
4. Why does vectorized code not necessarily use little memory?
5. Why are raw pixels a weak representation of semantic similarity?

## Source

- Stanford CS231n Spring 2025, Lecture 2: Image Classification with Linear
  Classifiers, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
