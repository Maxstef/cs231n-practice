# Long-range video modeling

Local 3D kernels model nearby frames well, but a label may depend on events far
apart in time. Several model families extend the temporal context.

## Common approaches

- **CNN plus RNN:** a CNN extracts frame features and an RNN processes their
  temporal sequence.
- **Non-local block:** every space-time position can aggregate information from
  every other position using self-attention.
- **Video Transformer:** a clip becomes a sequence of space-time tokens, which
  are processed by Transformer blocks.

For a feature map with shape `(N, C, T, H, W)`, flattening space and time gives
$L=THW$ tokens. Full self-attention creates an $L \times L$ score matrix per
head, so its memory and score computation grow approximately as

$$
O((THW)^2).
$$

A non-local block often uses $1 \times 1 \times 1$ convolutions for query, key,
value, and output projections. These act like shared linear layers at every
space-time location without mixing neighboring positions; attention performs
the non-local mixing.

## Reducing the cost

- factorize attention into a spatial step and a temporal step;
- form **tubelet tokens** from small space-time blocks rather than individual
  pixels or features;
- pool or merge tokens as the network becomes deeper;
- restrict attention to local windows or selected positions.

Factorized attention does not simply compute two identical attentions. Spatial
attention connects locations within a frame, while temporal attention connects
times for corresponding spatial positions. Their outputs are applied in stages
or combined according to the architecture.

## Key takeaway

Convolution supplies efficient local structure; attention supplies flexible
long-range interaction. Practical video models often combine them or reduce
the token count rather than using unrestricted attention at full resolution.

## Related practice

- Notebook 48: non-local blocks and spatiotemporal attention
- `cs231n_practice/video.py`

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
