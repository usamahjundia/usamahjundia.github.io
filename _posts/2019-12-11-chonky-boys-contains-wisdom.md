---
published: false
---
## What is hidden inside the weights of a randomly initialized neural network?

Earlier this week, a researcher i followed on twitter retweeted [this thread](https://twitter.com/Mitchnw/status/1201575787100561408) written by another researcher :

PLACEHOLDER

The paper mentioned can be accessed through [This link](https://arxiv.org/abs/1911.13299). After reading the abstract and eventually the whole paper, i found the idea to be pretty cool and i wanted to give it a go myself.

**TL;DR of the paper : Chonky bois (wide, randomly initialized neural nets) contains wisdom (a subnetwork that performs well on a given task)**

The main idea of the paper is that, hidden inside a randomly initialized neural network is a subnetwork that can reach good performance on a specified task, similar if not better in performance with a trained model.

The idea of optimizing a neural network without changing the weights has been explored in many papers, the most recent i read (and also mentioned in the paper behind this post) is the [Weight Agnostic Neural Networks](https://weightagnostic.github.io/) (WANN) which performs architecture search on a fixed value for all weights to reach good performance on certain tasks. But i digress. Probably in the future i will try to implement that too because frankly it is a cool idea.





