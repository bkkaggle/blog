---
toc: true
layout: post
description: Writing down my notes on memory usage in neural networks
categories: []
comments: true
hide: true
---
---

## Purpose

---

The purpose of these series of blog posts is to be a place to store my (still in-progress!) notes about topics in learning, help me keep track of everything I've learned over the last three years, and to practice my Latex skills.

This is my fourth blog post in the series, and this time I'm (again) really just Cmd+C'ing and Cmd+V'ing over some of my notes on memory usage in neural networks

---

## Background

---

GPU memory is used in a few main ways:

-   Memory to store the network's parameters
-   Memory to store the network's gradients
-   Memory to store the activations of the current batch
-   Memory used by optimizers (momentum, adam, etc) that stores running averages

---

### Parameter memory

---

Parameter memory usage depends on two things: The number of parameters and the amount of bytes used for each parameter.

-   Float 32 => 4 bytes
-   Float 16 => 2 bytes

You can calculate parameter memory with the formula:

```
parameter_memory = n_parameters * bytes_per_parameter
```

---

### Optimizer memory

--

-   SGD doesn't store any extra memory
-   Momentum doubles parameter memory usage by storing one momentum parameter per parameter in a model
-   Adam stores 2 momenum parameters so Adam uses 3x the parameter memory of SGD

#### Adafactor:

-   By default adafactor uses n+m memory to store momentum parameters for a nxm matrix
-   If you enable Beta_1, it will be like using momentum + n+m memory, so a little more than 2x SGD