---
toc: true
layout: post
description: writing down my notes on the different types of normalization used in neural networks
categories: [notes]
comments: true
---

> Updated on Jun 26, 2020: Fix BatchNorm and LayerNorm equations

---

## Purpose

---

The purpose of these series of blog posts is to be a place to store my (still in-progress!) notes about topics in machine learning, help me keep track of everything I've learned over the last three years, and to practice my Latex skills.

I'm starting this series of blog posts by writing down my notes on the different types of normalization in neural networks. Let's see how this goes.

---

## Why normalize?

---

Not normalizing input activations means that layers can transform activations to have very large or small means and standard deviations and cause the gradients to explode or vanish.

---

## Batch Normalization

---

[Arxiv](https://arxiv.org/abs/1502.03167)

**Tl;dr**: Calculate the mean and standard deviation for each feature in the batch across the batch dimension and normalize to have a mean of $0$ and a standard deviation of $1$, then scale the resulting activations by two learned parameters $\gamma$ and $\beta$

---

$$
\text{For a mini-batch of activations} \space B = { \{ x_{1} ... x_{m} \} },
$$

$$
\mu_{B} \leftarrow \frac{1} {m} \sum_{i=1}^{m} x_{i}
$$

$$
\sigma_{B}^{2} \leftarrow \frac{1} {m} \sum_{i=1}^{m} (x_{i} - \mu_{B}) ^ 2
$$

$$
\hat{x} \leftarrow \frac {x_{i} - \mu_{B}} {\sqrt {\sigma_{B}^{2} + \epsilon}}
$$

$$
y_{i} \leftarrow \gamma \hat{x_{i}} + \beta
$$

In Batch Normalization [^1], you first calculate the mean and variance of the input tensor across the batch dimension, then subtract the input tensor by the mean $\mu_{B}$ and divide by the standard deviation (plus a small value to prevent dividing by $0$) $\sqrt {\sigma_{B}^{2}}$ to restrict the activations of the neural network to having a mean of $0$ and a standard deviation of $1$

You then scale the activations with learned parameters by rescaling the zero-mean activations by two learned parameters $\beta$ and $\gamma$.

The original paper claimed that the reason batch norm worked so well was by reducing **internal covariate shift** ("The change in the distribution of the input values to a learning algorithm" [link](https://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)), but more recent papers have disputed this and given other reasons to why it works so well.

This lets the network choose the mean and standard deviation that it wants for its activations before they are passed to a convolutional or fully connected layer.

One question that I've had over and over again related to batch norm is where exactly to place it in a network, and it looks like other people [have](https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989/2) [had](https://forums.fast.ai/t/where-should-i-place-the-batch-normalization-layer-s/56825) [the](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/) [same](https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout) [  question](https://github.com/keras-team/keras/issues/1802).

The original paper places the batch norm layer after the convolutional layer and before the non-linearity, which is the default used by torchvision and other model zoos. It also claims that using batch norm can reduce or eliminate the need to use dropout, so the order could look like either of these:

$$
Conv \rightarrow BN \rightarrow ReLU \rightarrow Dropout
$$

$$
Conv \rightarrow BN \rightarrow ReLU
$$

Some benchmarks show that placing the batch norm layer after the non-linearity can perform better [^8]

$$
Conv \rightarrow ReLU  \rightarrow BN  \rightarrow Dropout
$$

but this isn't widely used.

One major disadvantage with this is that the pre-normalized activations must be saved for the backwards pass. This means that if you add a batchnorm layer for each convolutional layer in your network (which is a common practice), your network will need about twice the memory to store the same batch size into the GPU. Beyond using up more GPU memory, batch norm doesn't work with batch sizes of 1, and doesn't perform well with small batch sizes since the calculated mean and standard deviation for each batch will change a lot from batch to batch and gives the model a very noisy estimate of the true distribution.

Another thing you should keep in mind about batch norm is that when training on multiple gpus or machines, that by default, each gpu will keep its own mean and standard deviation parameters, which can be a problem if the per-gpu batch size is too low. There are synchronized batch norm implementations available that should fix this. Another thing to keep in mind is what mean and standard deviation values to use when evaluating on a test set or finetuning on a new dataset.

Other work, like **In-Place Batch normalization** [^2] reduces the memory usage by recomputing the pre-batchnorm activations from the post-batchnorm activations, while others, like **Fixup Initialization** [^3], **MetaInit** [^4], **LSUV** [^5], and **Delta Orthogonal** [^6] use special initialization strategies to remove the need for batch normalization.

---

## Layer normalization

---

[Arxiv](https://arxiv.org/abs/1607.06450)

**Tl;dr**: Calculate the mean and standard deviation for element in the batch across the feature dimension and normalize to have a mean of $0$ and a standard deviation of $1$, then scale the resulting activations by two learned parameters $\gamma$ and $\beta$

---

For activations in a batch of shape $x_{ij}$, where $i$ is the batch dimension and $j$ is the feature dimension (assuming this is a simple feedforward network),

$$
\mu_{i} \leftarrow \frac{1} {m} \sum_{j=1}^{m} x_{ij}
$$

$$
\sigma_{i}^{2} \leftarrow \frac{1} {m} \sum_{j=1}^{m} (x_{ij} - \mu_{i}) ^ 2
$$

$$
\hat{x} \leftarrow \frac {x_{ij} - \mu_{i}} {\sqrt {\sigma_{i}^{2} + \epsilon}}
$$

$$
y_{ij} \leftarrow \gamma \hat{x_{ij}} + \beta
$$

Layer Normalization [^7], is almost identical to batch normalization except that layer norm normalizes across the feature dimension instead of the batch dimension. This means that layer norm calculates a mean and standard deviation value for for each element in the batch instead of for each feature over all elements in the batch.

Layer norm is used mostly for RNNs and Transformers and has the same GPU memory requirements as batch norm.

---

## Resources

---

These are some of the amazing and very helpful blog posts, tutorials, and deep dives that have helped me learn about the topic and write this blog post.

-   https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/
-   https://github.com/pytorch/pytorch/issues/1959

---

## References

---

[^1]: https://arxiv.org/abs/1502.03167

[^2]: https://arxiv.org/abs/1712.02616

[^3]: https://arxiv.org/abs/1901.09321

[^4]: https://openreview.net/pdf?id=SyeO5BBeUr

[^5]: https://arxiv.org/abs/1511.06422

[^6]: https://arxiv.org/abs/1806.05393

[^7]: https://arxiv.org/abs/1607.06450

[^8]: https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
