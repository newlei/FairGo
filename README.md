# FairGo WWW2021

Learning Fair Representations for Recommendation: A Graph-based Perspective

![Overall_framework](figure/framework.jpg)

As a key application of artificial intelligence, recommender systems are among the most pervasive computer aided systems to help users find potential items of interests. Recently, researchers paid considerable attention to fairness issues for artificial intelligence applications. Most of these approaches assumed independence of instances, and designed sophisticated models to eliminate the sensitive information to facilitate fairness. However, recommender systems differ greatly from these approaches as users and items naturally form a user-item bipartite graph, and are collaboratively correlated in the graph structure. In this paper, we propose a novel graph based technique for ensuring fairness of any recommendation models. Here, the fairness requirements refer to not exposing sensitive feature set in the user modeling process. Specifically, given the original embeddings from any recommendation models, we learn a composition of filters that transform each user's and each item's original embeddings into a filtered embedding space based on the sensitive feature set. For each user, this transformation is achieved under the adversarial learning of a user-centric graph, in order to obfuscating each sensitive feature between both the filtered user embedding and the sub graph structures of this user. Finally, extensive experimental results clearly show the effectiveness of our proposed model for fair recommendation.
We provide PyTorch implementations for FairGo model.

## Prerequisites

- PyTorch
- Python 3.5
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/newlei/FairGo.git
cd FairGo
cd code
```
