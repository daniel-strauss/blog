---
title: Information Theory in LLM Theory
bibFile: data/bibliography.json # path relative to project root
---


## Introduction

## Background

 

### A Brief Introduction on Information Theory

In this section we want to provide a brief overview on Information Theory. 


Draft:
- start with everyday life example, that will make reader think of how one could formalize amount of information

- explain shannons information theory
    - name entropy definition
    - extract definition of information 
    - define mutual information, briefly mention why it is usefull


- briefly name an example applications from coding theory
- briefly name an example of neurology where neurons may maximize their entropy




### Philosofical Interlude: Does Information Exist?

Chapter Draft (everything in this draft is very vague, dont read)
- make reader know that you are critic of your own ideas
 - Previous Section information relied on an assumed probability space
 - what is probability? It is a way to formally express of what can be known and what can not be known? (<- express waeknasses of that thought and derive it and be more precise and present alternatives of expressong of what can be known and what not)
- Problems of probability:
    - How well can be known what can not be know? Well enough to express probabilitys?

    - maybe also everything can be known and everything happens with probability one (<- drop argument, why one can not be sure that universe is not deterministic universe, by using argument based on an explanation why even anything exists)
        - explain that under that circumstances only physically isolated can be known, using similar "proof" as holding theorems proof
        - include limitations of calculation power into probability?
- use previous arguments to express where probabilistic assumptions might  be off and name examples on how it could affect us practically
- conclude if we are carefull enough with probabilistic assumptions under scenario a everything has information 0, under scenario b everything has information $\infty$ and under scenario a using limitation of physical possible calculation power information content is impossibly hard to compute
- mention that information theory has been usefull anyway



## The Link of Information Theory and LLMs

## Analyzing Neural Network Architectures

In this chapter results from {{< cite "jeon2022information" >}} and {{< cite "jeon2024information" >}} will be discussed. For several generative models models upper bounds have been established of how much information they can generate. These results can be used make statements about the expressiveness of different neuronal architecture. (<- be more precise about expressivemness)


## An Information Theoretic Perspective Analysis on In-Context Learning

In {{< cite "jeon2024information" >}} assumptions about the origin of the trianing data of LLMs have been made from which an explanation of in context learning was stated.


### What is In-Context Learning


### Analysis of {{< cite "jeon2024information" >}}

### Discusion of Their Assumptions


## Conclusion

In this blog post several applications of Information Theory in Deep Learning and LLMs where presented. 


# References
{{< bibliography >}} 
