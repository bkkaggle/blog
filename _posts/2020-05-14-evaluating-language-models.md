---
toc: true
layout: post
description: The way you evaluate your language model can have a pretty big effect on validation loss and ppl values. Everyone should clearly report how their language models have been evaluated and try to evaluate their language models similarly to make comparing them easy.
categories: [ML, NLP]
comments: true
hide: true
---

## Tl;dr

The way you evaluate your language model can have a pretty big effect on validation loss and ppl values. Everyone should clearly report how their language models have been evaluated and try to evaluate their language models similarly to make comparing them easy.

---

This post is going to be a little different from my previous two posts, where I stuck to making posts to write down what I've learned about ML. This time, I'm still making notes, but I'll also be writing about my work [trying](https://github.com/huggingface/transformers/issues/483) [to](https://github.com/openai/gpt-2/issues/78) [replicate](https://github.com/huggingface/transformers/issues/491) GPT-2's zero-shot results on wikitext2 and wikitext103.

I'm currently working on finetuning gpt2-like models on small datasets and I wanted to compare the results of my finetuned models on wikitext2 to OpenAI's baseline zero-shot results. This sounded like a pretty easy thing to do, but there are many ways that the authors of different papers choose to evaluate and compare their language models—and not all of them are easily comparable.

Different factors can have an impact on the val or test perplexity for a language model on a particular dataset—The **vocabulary size** of your language model, the **context length** that you use to evaluate on, and your **evalutation method** can all make a big difference.

---

## Vocabulary size

---

The size of the input vocabulary for your language model can make it easier or harder for your language model to predict the next token in a sequence, for example, a character level language model with 26 tokens (one for each letter of the english alphabet) will have a lower perplexity that a word level language model with hundreds of thousands of tokens. Think of it like this, it's a lot easier to predict the next letter in the sentence `I’m a computer science and machine learning enthusias` (which would be the letter `t`) than the next word in the sentence `I'm a computer science and machine learning` (which is the word `enthusiast`). This would mean that a character-level language model would have a much lower perplexity value than a word-level model, and that you may be able to break SOTA on most language modelling datasets by just changing the vocabulary!

To make sure that models trained on using different tokenizers (word-level, character-level, BPE, etc) can be compared, you can normalize the loss of a language model with a vocabulary of $V_1$ tokens to a common vocabulary of $V_2$ tokens by multiplying the average loss of the language model with vocabulary size $V_1$ by the ratio between $V_1$ and $V_2$ (you could also sum the losses from the all the tokens and then divide by the number of tokens, but since the two give the identical result, I'll just refer to the version where we take the average loss):

$$
normalized\_loss = loss_{V_1} * \frac {V_1} {V_2}
$$

---

## Context Length and Evaluation Method

---

Language models compute the probability of a sequence $s$ with $n$ tokens with:

$$
P(s) = \prod_{i = 1}^{n} p(w_i | w_1 ... w_{i-1})
$$

Datasets can have thousands to millions to hundreds of millions of tokens, so sending the entire dataset to the language model at once isn't possible. To make the calculation of the loss and the perplexity computationally possible, there are two approaches that I've seen other people use:

1. Splitting the dataset into chunks of length `context_len`, passing each chunk to the lm separately, and averaging the loss over all the chunks

2. Using an overlapping sliding windows approach, still only passing chunks of length `context_len` to the model at a time, but overlapping $t$ tokens from the previous sequence and not counting these overlapped tokens when calculating the loss.

Approach #1 is the easiest to implement (Like in the official Pytorch example, your dataset would just load in the validation text file, tokenize it, and break it up into `context_len` chunks to be iterated over) but isn't optimal since the lm won't have any context to use when predicting the first token in each batch.

This is also the approach taken by most tutorials and reference implementations for evaluating language models. For example, the Pytorch examples for word-level language modelling on wikitext-2 [^1], the AWD-LSTM repository [^2], and the /transformers library's language modelling example [^3] all evaluate on fixed chunks of length `context_len`.

In contrast, approach #2 is used by Transformer-XL [^4] and Megatron-LM [^5] and is a little more difficult to implement, you still need to break the tokenized validation file into chunks of length `context_len` but only move the start of each chunk $t$ tokens ahead at a time. The value of $t$ that you choose will make a difference, if you priorize the precision of the resulting loss value and set $t = 1$, your loss will be closer to the true value over the dataset than if you choose $t = 30$ (like Megatron-LM), but using a lower value of $t$ will also increase the amount of time it will take to calculate the loss over the entire validation set, especially if it is very large. Using overlapping sliding windows also means that you will have to only count the loss of the non-overlapping segments, masking out the loss for the first $t$ tokens. The Transformer-XL [^4] paper discusses this topic in section 3.1 and shows how its cached sequence of hidden states from previous timesteps lets it evaluate on overlapping sliding windows at a lower computational cost.

Whichever approach you choose, the value of `context_len` that you choose will also make a significant effect on your loss. On my experiments with gpt2, I could see a decrease of 4ppl across many model sizes (gpt2-medium, gpt2-large, gpt2-xl) just by increasing the context len that the models were evaluated on from 256 to 1024.

---

## GPT-2 and zero-shot results on wikitext2 and wikitext103

---

OpenAI's GPT-2 [^6] paper is pretty short on details when it comes to how they ran zero-shot (no finetuning!) evaluation on a range of datasets and several people have also had some trouble [trying](https://github.com/huggingface/transformers/issues/483) [to](https://github.com/openai/gpt-2/issues/78) [replicate](https://github.com/huggingface/transformers/issues/491) their results.


| model-size  | loss on wikitext103's test set | perplexity | adjusted perplexity | reported perplexities |
| ----------- | ------------------------------ | ---------- | ------------------- | --------------------- |
| gpt2        | 3.149                          | 23.33      | 35.12               | 37.5                  |
| gpt2-medium | 2.923                          | 18.59      | 27.18               | 26.37                 |
| gpt2-large  | 2.786                          | 16.23      | 23.30               | 22.05                 |
| gpt2-xl     | 2.706                          | 14.97      | 21.28               | 17.48                 |

I was able to get these results on WikiText-103's test set that are pretty close (except for gpt2-xl, that's off by almost 4ppl) to the paper's reported results after a bit of experimenting, here's what I did:

For my zero-shot results, I used a non-overlapping context length of 1024 tokens (using overlapping sliding windows should get you better results and get you to OpenAI's results). As for adjusting the loss to account for GPT-2's custom tokenizer, I used the normalized loss calculation from above with the original and tokenized number of tokens from the test file—I split the preprocessed test set on spaces to get $217646$ tokens, and with GPT-2's tokenizer to get $249612$ tokens.

OpenAI says in section 3.1 that they used invertible detokenizers to remove tokenization artifacts from the processed WikiText-103 test set (`wiki.test.tokens`) (like extra spaces before and after punctuation marks) created by the original authors of the dataset. Since they didn't provide details on what preprocessing artifacts they removed in either the paper or code, I used the Megatron-LM [^5] project's [invertible detokenizers](https://github.com/NVIDIA/Megatron-LM/blob/master/tasks/zeroshot_gpt2/detokenizer.py) that they used for their own zero-shot evaluation results on WikiText-103.

---

## References

---

[^1]: https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L136

[^2]: https://github.com/salesforce/awd-lstm-lm/blob/master/finetune.py#L104

[^3]: https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py

[^4]: https://arxiv.org/abs/1901.02860

[^5]: https://arxiv.org/abs/1909.08053

[^6]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf