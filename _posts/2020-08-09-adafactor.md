---
toc: true
layout: post
description: Writing down my notes on the adafactor optimizer
categories: [notes]
comments: true
---

---

## Purpose

---

The purpose of these series of blog posts is to be a place to store my (still in-progress!) notes about topics in machine learning, help me keep track of everything I've learned over the last three years, and to practice my Latex skills.

This is my fifth blog post in the series, and this time I'm taking some notes on the Adafactor optimization paper

---

## Adam default hyperparmeters

---

-   $\beta_1 = 0.9$
-   $\beta_2 = 0.999$
-   linear warmup + inv sqrt decay

-   The default/initial lr for most experiments (the ones with a step size of $a_t = 0.1 * s_t$) is $1e-3$.
-   The authors use an inverse sqrt learning rate decay schedule for all experiments

-   warmup helps but not 100% necessary

---

## Adafactor

---

-   Arxiv: https://arxiv.org/abs/1804.04235

Adafactor factorizes the second moment running averages of the gradient into row and column vectors. The matrix is then "divided by the sum of all entries" in the matrix to approximate the original matrix (Section 3)

By default, Adafactor doesn't work if you don't use a learning rate warmup. The authors tried using either only the row or column running averages. Using only the row running averages works almost just as well, but using only the column running averages doesn't work at all.

_Note: check if you can get away with only using row means without warmup_.

Also, most of the Adafactor benchmarks in the paper were done with $\beta_1 = 0$, but IIRC some pretraining papers use it (?).

---

## The problem with Adam/Adafactor (section 5)

---

Adam without $\beta_1$ works almost just as well as the original Adam implementation and it saves you quite a bit of memory. Note that this only holds true when you're using a linear warmup with it. The problem is that using a fast ($0.9$) $\beta_2$ leads to Adam not converging no matter what, while using a slow ($0.999$) $\beta_2$ leads to your model only training well if you also use warmup\_.

**You either need a $\beta_1$ of $0.9$ or warmup with a $\beta_2$ of $0.999$.**

Here's why: Using a slow $\beta_2$ means that second moment information is updated very slowly, leading to the current value of the running average matrix be out of date (this is shown in section 6, figure 1).

How do we fix this?

Well, the authors outline a few ways on how to do so...

---

### 1. Gradient clipping (section 6)

---

Having an out of date second moment estimator means that the raw gradient updates are often larger than they should be. A simple way to fix this would be to just scale down the magnitude of the update if it is larger than a particular "clipping" value. Empirically, update clipping works well when training without warmup but doesn't match the original's performance. The authors show that clipping at $1$ works well with both Adam and Adafactor.

_This is referred to in the paper as clipping, which it technically is, but acts more like gradient scaling since you're really only scaling down the magnitude of the gradient update when it passes a particular "clipping" threshold._

---

### 2. Gradually increasing $\beta_2$

---

Add a schedule ($1 - t ^ {- x}$) that gradually increases the $\beta_2$ from $0$ to $1$. The quality of the results for when you're training without warmup are very dependent on the value of $x$ that you choose. It seems like it stabilizes when you use this with update clipping, but the end result of using a $\beta_2$ schedule + update clipping is really no better than just update clipping.

_Does this mean that a $\beta_2$ schedule is practically useless?_

---

### 3. Relative update size

---

Instead of hardcoded learning rate, multiply the gradient update by "the root-mean-square of its components, lower-bounded by a small constant 2". In equation form (taken from Section 9, algorithm 4), it's

$$
\epsilon_2 = 1e-3
$$

$$
p_t = \max(\epsilon_2, RMS(X_{t - 1}))
$$

In practice, the authors combine this with $\beta_2$ scheduling and update clipping.

_It would be nice to see how the relative update size method performs by itself without the scheduler or update clipping but 🤷‍♂️_

The authors try adding a $\beta_1$ of $0.9$, but that actually makes the results slightly worse.

---

## Conclusion

---

Most codebases that I've seen (this includes all of mine too!) use the Adafactor optimizer don't use it the way that the authors reccommend to use it in their paper. It's pretty common to see people use Adafactor without a $\beta_1$, without the $\beta_2$ decay schedule, and with a simple linear warmup and decay.

For my future self looking back at this post to figure out what hyperparmeters they should use for Adafactor (or anyone else who's reading this), here's a summary for what hyperparameters to use with Adafactor:

-   no warmup
-   no $\beta_1$
-   Adafactor's built-in inv sqrt lr decay
-   update clipping at $1.0$
-   Relative update step sizes instead of a fixed learning rate