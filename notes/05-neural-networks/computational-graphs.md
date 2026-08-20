# Computational graphs

A computational graph represents a calculation as a sequence of small operations.
It provides two views of the same computation:

- the **forward pass** computes intermediate values and the final output;
- the **backward pass** computes how the loss changes with respect to each value.

Breaking a large expression into simple operations makes its derivatives easier to
derive, implement, and test.

## A small scalar example

Consider

$$
q = x + y, \qquad f = qz.
$$

For $x=-2$, $y=5$, and $z=-4$, the forward pass gives

$$
q=3, \qquad f=-12.
$$

The backward pass starts at the output with

$$
\frac{\partial f}{\partial f}=1.
$$

At the multiplication node,

$$
\frac{\partial f}{\partial q}=z=-4,
\qquad
\frac{\partial f}{\partial z}=q=3.
$$

The addition node has local derivatives equal to one. By the chain rule,

$$
\frac{\partial f}{\partial x}
=\frac{\partial f}{\partial q}\frac{\partial q}{\partial x}
=-4,
$$

and similarly

$$
\frac{\partial f}{\partial y}=-4.
$$

This illustrates the central backpropagation rule:

> gradient flowing into a node $\times$ the node's local derivative.

The first factor is often called the **upstream gradient**. The result sent toward
an input is its **downstream gradient**.

## Common backward patterns

### Addition

For $z=x+y$, both inputs receive the upstream gradient unchanged:

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z},
\qquad
\frac{\partial L}{\partial y}=\frac{\partial L}{\partial z}.
$$

### Multiplication

For $z=xy$, each input receives the upstream gradient multiplied by the other
input:

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}y,
\qquad
\frac{\partial L}{\partial y}=\frac{\partial L}{\partial z}x.
$$

### Maximum and ReLU

A maximum operation routes the gradient through the input that won during the
forward pass. ReLU is the elementwise case $\max(0,x)$: positive inputs pass the
gradient and negative inputs block it.

### Branching

If one value is used by several later operations, it influences the loss through
several paths. Its gradient is the **sum** of the contributions from all paths.

A useful memory rule is:

- multiply derivatives while moving along one path;
- add gradient contributions when paths meet.

## Why cache forward-pass values?

Backward formulas often need values from the forward pass. A multiplication node
needs both inputs; ReLU needs to know which inputs were positive. A layer can cache
these values during `forward` and reuse them during `backward`.

Backpropagation processes nodes in reverse dependency order: a node is processed
after all gradient contributions from later nodes have arrived.

## Connection to this project

- Notebook 12 introduces computational graphs and local derivatives.
- `cs231n_practice/gradient_check.py` compares analytical gradients with numerical
  approximations.
- `cs231n_practice/layers.py` packages forward computations and their matching
  backward computations into reusable functions.

## Key takeaway

A computational graph does not change the mathematics. It organizes the chain rule
so intermediate results are reused and each operation only needs its own local
derivative.
