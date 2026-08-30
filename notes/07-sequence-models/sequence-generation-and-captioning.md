# Sequence generation and image captioning

Sequence generation connects recurrent states to discrete token predictions.
The same core pieces support character-level language modeling and
image-conditioned caption generation.

## Tokens and embeddings

A vocabulary maps each symbol to an integer ID. IDs are discrete lookup
addresses, not meaningful numerical measurements. An embedding matrix

$$
E\in\mathbb{R}^{V\times D}
$$

stores one learned vector for each of $V$ tokens. For token IDs with shape
$(N,T)$, lookup produces embeddings with shape $(N,T,D)$.

When a token occurs repeatedly, every occurrence selects the same row of $E$.
Its backward gradient must therefore accumulate all contributions into that
shared row.

## From hidden states to vocabulary scores

An RNN returns hidden states with shape $(N,T,H)$. A temporal affine projection
maps each state into $V$ scores:

$$
S_{n,t,:}=h_{n,t,:}W_{out}+b_{out},
$$

where $W_{out}:(H,V)$ and $S:(N,T,V)$. The projection uses the same parameters at
every position and does not mix time steps; temporal context is already encoded
in the recurrent states.

Softmax converts one score vector into a distribution over the next token.
During training, temporal cross-entropy adds the valid position losses. Padding
positions are masked so they contribute neither loss nor score gradient.

## Training alignment

For a stored caption

```text
<START>  a  cat  sleeps  <END>  <PAD>
```

the shifted arrays are

```text
input:   <START>  a      cat     sleeps  <END>
target:  a        cat    sleeps  <END>   <PAD>
mask:    True     True   True    True    False
```

The mask follows the **targets**, because the loss asks whether each next-token
target is real or padding. Predicting `<END>` is a real learning task;
predicting `<PAD>` is not.

Training usually uses **teacher forcing**: the correct previous token is
available as input at every position. This permits batched computation of the
known input sequence, even though the recurrent hidden states themselves are
still computed in time order.

## Autoregressive inference

At inference time the future caption is unknown:

1. provide `<START>` as the first input;
2. compute the next-token scores;
3. choose or sample one token;
4. feed that predicted token back as the next input;
5. stop at `<END>` or a length limit.

Generation is sequential because each new input depends on the preceding
prediction. Greedy decoding chooses the largest score at every step. It is fast
but can miss a better complete sequence because it never revisits an earlier
choice. Sampling produces more varied sequences; beam search keeps several
partial candidates.

Special tokens such as `<START>` and `<PAD>` can be forbidden in the output by
setting their decoding scores to negative infinity before selection.

## Image-conditioned captioning

A CNN or vision backbone first converts each image into a feature vector
$F:(N,D_{image})$. One simple recurrent captioning design projects this vector
into the initial hidden state:

$$
h_0=FW_{image}+b_{image}.
$$

The caption pipeline is then

```text
image features -> h0
caption IDs -> embeddings -> RNN hidden states
hidden states -> vocabulary scores -> masked next-token loss
```

During backpropagation, the loss gradient reaches the image projection through
$dh_0$. If the vision backbone is trainable, the resulting feature gradient can
continue into it; if features are precomputed, only the captioning parameters
are updated.

At sampling time, image features establish the initial context and `<START>`
begins generation. An untrained model produces arbitrary tokens because it has
not learned how image content, caption prefixes, and next words are related.

## RNN captioning limitation

Compressing the entire image into one $h_0$ vector creates an information
bottleneck. Attention-based captioning instead lets each decoding step consult
spatial image features directly. This provides the bridge from recurrent
captioning to the next topic: attention and Transformers.

