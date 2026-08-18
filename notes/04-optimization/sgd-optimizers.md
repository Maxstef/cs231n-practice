# SGD optimizers

An optimizer converts a gradient estimate $g_t$ into a parameter update. The
simplest rule is vanilla SGD:

$$
W_{t+1} = W_t - \alpha g_t.
$$

## Why vanilla SGD can struggle

- **Noisy gradients:** different minibatches point in somewhat different
  directions.
- **Poor conditioning:** the loss can change steeply in one direction and
  slowly in another. SGD then oscillates across the steep direction while
  progressing slowly along the shallow one.
- **Small-gradient regions:** gradients can become small near plateaus, saddle
  points, or local minima. In high-dimensional neural-network objectives,
  saddle points are an especially important concern.

These issues motivate update rules that remember previous gradients or adapt
the step size for each parameter.

## SGD with momentum

Momentum accumulates a running direction $v_t$:

$$
v_{t+1} = \rho v_t + g_t,
$$

$$
W_{t+1} = W_t - \alpha v_{t+1}.
$$

The momentum coefficient $\rho$ determines how strongly previous gradients are
retained; values such as $0.9$ are common starting points. Directions that stay
consistent accumulate speed, while opposing oscillations partly cancel. This
often reduces jitter and helps move through shallow regions.

## RMSProp

RMSProp tracks an exponential moving average of elementwise squared gradients:

$$
s_t = \rho s_{t-1} + (1-\rho)g_t^2,
$$

and uses it to scale each parameter's update:

$$
W_{t+1} = W_t - \alpha\frac{g_t}{\sqrt{s_t}+\epsilon}.
$$

All products, square roots, and divisions here are elementwise. A parameter
that repeatedly receives large gradients gets a smaller effective step;
one with smaller gradients gets a relatively larger step. The small
$\epsilon$ prevents division by zero.

## Adam

Adam combines a momentum-like first moment with an RMSProp-like second moment:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2.
$$

Because both estimates start at zero, early values are biased toward zero.
Adam corrects this:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

The update is

$$
W_{t+1} = W_t - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
$$

Typical starting values are $\beta_1=0.9$ and $\beta_2=0.999$, but the learning
rate still needs attention.

## Practical summary

- **Vanilla SGD:** simplest baseline; sensitive to learning rate and loss
  geometry.
- **Momentum:** smooths noisy directions and builds speed in persistent ones.
- **RMSProp:** gives parameters adaptive effective step sizes.
- **Adam:** combines momentum, adaptive scaling, and bias correction.

No optimizer guarantees a good model. Data preprocessing, correct gradients,
initialization, regularization, batch size, and learning-rate choices still
matter. Optimizers also follow different trajectories, so their learning rates
should not be compared as if the numeric values were interchangeable.
