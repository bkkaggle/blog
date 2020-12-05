---
toc: true
layout: post
description: Writing down some of my notes on initializing neural networks 
categories: [notes]
comments: true
hide: true
---

---

## Purpose

---

The purpose of these series of blog posts is to be a place to store my (still in-progress!) notes about topics in learning, help me keep track of everything I've learned over the last three years, and to practice my Latex skills.

This is my third blog post in the series, and this time I'm really just Cmd+C'ing and Cmd+V'ing over some of my notes on initialization for neural networks

---

## Notation

---

-   $\mu$: Mean
-   $\sigma ^ 2$: Standard Deviation
-   $c\_{in}$: Number of input channels to a layer
-   $c\_{out}$: Number of output channels to a layer

---

## Initialization

---

-   $\mu$ and $\sigma ^ 2$ of activations should be close to $0$ and $1$ to prevent the gradients from exploding or vanishing
-   activations of layers have $\sigma ^ 2$ close to $\sqrt {c\_{in}}$
-   so, to get the $\sigma ^ 2$ back to $1$, multiply randomly initialized weights by $1 / sqrt(c\_{in})$
-   this works well without activations, but results in vanishing or exploding gradients when used with a tanh or sigmoid activation function
-   bias weights should be initialized to $0$
-   intializations can either be from a uniform distribution or a normal distribution
-   use **Xavier Initialization** for sigmoid and softmax activations
-   use **Kaiming Initialization** for ReLU or Leaky ReLU activations

---

### Xavier or Glorot Initialization

---

#### Uniform initialization:

---

-   bound a uniform distribution between $\pm \sqrt { \frac {6} {c\_{in} + c\_{out}}}$

---

#### Normal initialization:

---

-   multiply a normal distribution by $\sqrt \frac {2} {c\_{in} + c\_{out}}$
-   or create a normal distribution with $\mu = 0$ and $\sigma ^ 2 = \sqrt \frac {2} {c\_{in} + c\_{out}}$
-   helps keep identical variances across layers

---

### Kaiming or He initialization

---

-   when using a ReLU activation, $\sigma ^ 2$ will be close to $\sqrt \frac {c\_{in}} {2}$, so multiplying the normally distributed activations by $\sqrt \frac {2} {c\_{in}}$ will make the activations have a $\sigma ^ 2$ close to $1$

---

#### Uniform initialization:

---

-   bound a uniform distribution between $\pm \sqrt \frac {6} {c\_{in}}$

---

#### Normal initialization

---

-   multiply a normal distribution by $\sqrt \frac {2} {c\_{in}}$
-   or create a normal distribution with $\mu = 0$ and $\sigma ^ 2 = \sqrt \frac {2} {c\_{in}}$

---

### Gain

---

-   multiplied to init bounds/stddevs
-   $\sqrt 2$ for ReLU
-   none for Kaiming

---

### Pytorch defaults

---

-   most layers are initialized with Kaiming uniform as a reasonable default
-   use Kaiming with correct gain (https://pytorch.org/docs/stable/nn.html#torch.nn.init.calculate_gain)

---

### Resources

---

-   https://github.com/pytorch/pytorch/issues/15314
-   https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c
-   https://pytorch.org/docs/stable/_modules/torch/nn/init.html
-   https://discuss.pytorch.org/t/whats-the-default-initialization-methods-for-layers/3157/21
-   https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
-   https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
-   https://pytorch.org/docs/stable/nn.html#torch.nn.init.calculate_gain
-   https://github.com/mratsim/Arraymancer/blob/master/src/nn/init.nim
-   https://jamesmccaffrey.wordpress.com/2018/08/21/pytorch-neural-network-weights-and-biases-initialization/