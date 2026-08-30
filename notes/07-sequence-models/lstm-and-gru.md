# LSTM and GRU gated recurrent cells

Vanilla RNNs compress all recurrent information into one hidden state and
replace that state through a nonlinear transformation at every step. LSTMs and
GRUs introduce gates that learn when information should be retained, updated,
or exposed.

## LSTM has two recurrent states

An LSTM carries both:

- hidden state $h_t$: the exposed representation used by outputs and the next
  gate calculation;
- cell state $c_t$: an internal memory path updated additively.

Both $h_{t-1}$ and $c_{t-1}$ enter an LSTM step. The previous hidden state and
current input first produce four vectors at once:

$$
A=x_tW_x+h_{t-1}W_h+b,
$$

where $A$ has shape $(N,4H)$. Splitting its final axis into four blocks of width
$H$ gives the gate pre-activations. A common convention is

$$
i=\sigma(A_i),\qquad
f=\sigma(A_f),\qquad
o=\sigma(A_o),\qquad
g=\tanh(A_g).
$$

The four blocks are not four input arrays. They are four learned projections
packed into one larger matrix multiplication for efficiency.

## Meaning of the gates

- **Input gate $i$:** how much candidate information to write.
- **Forget gate $f$:** how much previous cell memory to retain.
- **Output gate $o$:** how much cell information to expose as hidden state.
- **Candidate $g$:** proposed new memory content. It is commonly called the
  candidate rather than a gate because `tanh` supplies content in $[-1,1]$.

The state updates are

$$
c_t=f\odot c_{t-1}+i\odot g,
$$

$$
h_t=o\odot\tanh(c_t),
$$

where $\odot$ denotes element-wise multiplication.

The cell state appears **after** the four blocks are computed: the gates use
$x_t$ and $h_{t-1}$ to decide how to combine old cell memory $c_{t-1}$ with the
new candidate $g$.

## Why the cell path helps gradients

The cell update contains an additive path, and its direct derivative is

$$
\frac{\partial c_t}{\partial c_{t-1}}=f.
$$

When the forget gate stays near one, gradients can travel across several cell
updates without repeatedly passing through a full `tanh` transformation and
recurrent matrix multiplication. This resembles the benefit of an identity or
residual path.

This does not guarantee perfectly unchanged gradients: the product of forget
gates can still shrink, and other gradient paths still pass through nonlinear
operations. LSTMs make preservation learnable rather than automatic.

## Backward flow through the cell

An LSTM backward step receives gradients for both outputs:

$$
dh_t=\frac{\partial L}{\partial h_t},\qquad
dc_t^{future}=\frac{\partial L}{\partial c_t}\text{ from the next step}.
$$

Because $h_t$ also depends on $c_t$, the total cell gradient is

$$
dc_t=dc_t^{future}
    +dh_t\odot o\odot\left(1-\tanh^2(c_t)\right).
$$

This gradient branches into the retained-memory and new-memory paths:

$$
dc_{t-1}=dc_t\odot f,
$$

$$
df=dc_t\odot c_{t-1},\qquad
di=dc_t\odot g,\qquad
dg=dc_t\odot i.
$$

The gate gradients then pass through their sigmoid or `tanh` derivatives and
through the packed affine transformation.

## GRU comparison

A gated recurrent unit usually carries only one hidden state. Its update gate
interpolates between the previous state and a candidate:

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\widetilde{h}_t.
$$

A reset gate controls how much previous state participates in the candidate.
Compared with an LSTM, a GRU has:

- no separate cell state;
- fewer gates and parameters;
- a simpler state update;
- the same broad goal of providing selective memory and easier gradient paths.

Neither cell is universally superior. Their suitability depends on the data,
sequence length, compute budget, and optimization behavior.

## Shape reminder

For input dimension $D$ and hidden dimension $H$:

```text
x_t       (N, D)
h_prev    (N, H)
c_prev    (N, H)       # LSTM only
W_x       (D, 4H)      # LSTM packed gates
W_h       (H, 4H)
b         (4H,)
h_next    (N, H)
c_next    (N, H)
```

