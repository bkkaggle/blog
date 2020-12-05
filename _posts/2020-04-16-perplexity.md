---
toc: true
layout: post
description: I'm taking notes on evaluation metrics in NLP 
categories: [notes]
comments: true
---

> Updated on Aug 2, 2020: Add link to more resources

---

## Purpose

---

The purpose of these series of blog posts is to be a place to store my (still in-progress!) notes about topics in learning, help me keep track of everything I've learned over the last three years, and to practice my Latex skills.

This is my second blog post in the series, and this time I'm taking notes on evaluation metrics in NLP.

Most of the content of this post comes from [Chip Huyen's](https://huyenchip.com/) really good article in [The Gradient](https://thegradient.pub/) on [Evaluation methods for language models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/) and the [Deep Learning](https://www.deeplearningbook.org/) book, so a big thank you to the authors and editors for making this perplexing (pun intended) topic easy to understand.

> let me be 100% clear here, I don't want to come across like I'm taking someone else's ideas and publishing them as my own. The purpose of this blog post is to take notes for myself so I can come back to this when I inevitably forget how to calculate perplexity.

Also, take a look at [this](https://sjmielke.com/comparing-perplexities.htm) for another good look at perplexity and the effect of tokenization on it.

---

## Background

---

Language models like GPT2 try to predict the next word (or subword/character, we'll use the term `token` in this blog post), in a context of tokens.

For example, when predicting the next word in the sentence `"I am a computer science and machine learning"`, the probability of the next work being `enthusiast` could be represented by

$$
P(enthusiast | I \space am \space a \space computer \space science \space and \space machine \space learning)
$$

The probability of a sentence $s$, where $s$ is a sequence of n tokens $(w_{0}, w_{1}, ... w_{n})$ can be represented as

$$
P(s) = \prod_{i = 1}^{n} p(w_i | w_1 ... w_{i-1})
$$

expanded, it looks like this:

$$
P(s) = p(w_{1})p(w_{2} | w_{1})p(w_{3} | w_{1}, w_{2})...p(w_{n} | w_{1} w_{2} ... w_{n - 1})
$$

---

## Information Theory

---

The amount of information given by a discrete event $x$ is calculated by the **Self-Information** equation [^5] <a name="5"></a>

$$
I(x) = -log \space P(x)
$$

Information is normally written in one of two units, $nats$, in which case the logarithm has a base of $e$ or $bits$, with a base of $2$.

One $nat$ encodes the "amount of information gained by observing an event with a probability of $\frac {1} {e}$." [^5] <a name="5"></a>

---

## Shannon Entropy

---

**Shannon entropy** is the extension of the **Self-Information** equation to probability distributions and is a way to "quantify the amount of uncertainty in an entire probability distribution." [^5] <a name="5"></a>

$$
H(x) = \mathbb E_{x \sim P} [log \space P(x)]
$$

It's a measure of how much information, on average is produced for each letter of a language [^1] <a name="1"></a> and (if calculated in units of
$bits$) can also be defined as the average number of binary digits required to encode each letter in a vocabulary.

In NLP, the evaluation metric, **Bits-per-character** (BPC), is really just the entropy of a sequence, calculated with units of bits instead of nats.

Entropy calculated across language models that are trained over different context lengths aren't exactly comparable, LMs with a longer context len will have more information from which to predict the next token. For example, given the sentence `I work with machine learning` it should be easier for a LM to predict the next word in the sequence `I work with machine`, than with just the first word: `I`. _(This is actually a major pain point when I was trying to reproduce gpt2's ppl numbers on wikitext2 and wikitext103, it's still unclear how the paper evaluated the ppl values on the tests sets for both datasets.)_

---

## Perplexity

---

**Perplexity**: A measurement of how well a probability distribution or probability model predicts a sample [^2] <a name="2"></a>

Perplexity is usually calculated with units of $nats$, so calculate it with the equation: $PPL = e^{loss}$

---

## Dealing with different tokenization schemes

---

If you want to convert the perplexity between models that have been trained using different tokenization schemes and have a different number of tokens that the LM can predict, multiply the cross-entropy loss of the first language model by the ratio of $(\text{n tokens first model} / \text{n tokens seconds model})$

The adjusted perplexity value can be found with [^4] <a name="4"></a>:

$$
adj\_ppl = e^{loss * (\text{\#tokens} / \text{\#tokens for other model})}
$$

---

## References

---

[^1]: Claude E Shannon. Prediction and entropy of printed english. Bell system technical journal, 30(1):50–64, 1951.

[^2]: https://en.wikipedia.org/wiki/Perplexity

[^3]: https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc/261789

[^4]: https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L282

[^5]: Chapter 3, Deep Learning, Ian Goodfellow, Yoshua Bengio and Aaron Courville, 2016, MIT Press