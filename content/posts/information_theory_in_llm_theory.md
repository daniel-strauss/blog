---
title: Some Information Theoretic Perspectives on LLM Theory
bibFile: data/bibliography.json # path relative to project root
---

⚠️ 📥 😚 🛡 🚦 👹 🌿 *By reading this blogpost, you will find out, why this emoji sequence is here.*


TODO rewrite thetas, random theta: big, constant theta small

## Introduction

In this Blogpost you will learn about some papers, that explored LLMs through the lens of Information Theory. We will look at one paper in detail, that looked specifically at the phenomenon of in-context learning by making assumptions about the probability distribution of the training data.
Then we will carefully examine the assumptions made in that paper. To do that well we will proof some of our own statements, to get more insight into the validity of these assumptions.


### A Brief Introduction on Information Theory

![targets](/figures/venn_information.png "Assumptio Sparse Mixture") 


In this section we want to provide a brief overview on Information Theory. 

Imagine you lived $40\cdot n$ years in the future and that the population has doubled every $40$ years. For simplicity assume that in the current present just one person is alive. Conclusively $2^n$  people live at your point in the future. You live very far in the future and $n$ is very large $n=2^{10^{15}}$. "Today" is your friends birthday party and you dont have a present. 

Therefore you decide to hire last minute a mathgician for his party as mathagicians are know to let partys go wild with their magic and math. 
In order to hire the mathgician you have to put in your friends one petabit ($=10^15$ bits) long address into the website of the mathgician-firm, such that he can find the location of the party. (The address is one petabit long as in the future every inhabitant has his own address, and there are $n$ people.) At the party the mathgician shows up and everyone has a good time. But then suddenly you start to wonder:

You only have a 1Gbit/s upload speed (hardware did not improve so much in the last $40\cdot2^{10^{15}}$ years), how have you been able to upload $10^{15}$ bits (one PentaBit) within just one day? This seems impossible as it takes $10^6$ seconds to upload one PentaBit with your network speed. Then you find out why: the website did not have to send $10^{15}$ bits of information to its server. Small parts of the information have already been known by the server; For example the information that people in your region are more likely to order a magician for someone in the same galaxy was already present on their server. 
Could they have used that information to such that they required less bits of information from you, such that your computer had to send them a smaller amount of bits? How can something seeming as soft as information influence something as hard as a bit-sequence length? What is the math behind this?


Information theory is mainly the study on expressing the quantity of information called Entropy. Entropy of an information is the minimal expected amount of a quantity like bits required to encode this information.

One can show, that the entropy $\mathbb H$ in bits of a discrete random variable $X$ with a known probability density function $p_X$ is

$\mathbb H(X) = \sum_{x \in \mathcal X} - p_X(x) \log_2(p_X(x)) = \mathbb E[-\log_2 X]$.

Therefore the above term is usually referred as the formal definition of entropy. Often instead of the binary logarithm the natural logarithm is used to express entropy. The resulting unit is called nats instead of bits. 

$\mathbb H(X) = \sum_{x \in \mathcal X} - p_X(x) \log(p_X(x))$

The entropy of continoous random variables is infinity, as they can have infinetly many outcomes (TODO provide source). But the differential entropy $\bold h$ of a continous random variable. (TODO explain use of diferential entropy)


$\bold h(X) = \mathbb E[-\log(X)]$


Information theory provides several expressions for how the information of different random variables relate.  This is useful in many different scenarios. (<- TODO example)
Definition:

- Joint Entropy: $\mathbb H(X,Y) := \mathbb H((X,Y))$
- Conditional Entropy: $\mathbb H(X|Y) := \mathbb E[(y \to \mathbb H(X|Y=y))(Y)] = \sum_{y \in \mathcal Y} p_y(y) \sum_{x \in \mathcal X} p_{X|Y=y}(x) - \log(p_{X|Y=y}(x))$ (<- todo chec correct)
- Mutual Information: $\mathbb I(X;Y) := \mathbb H(X) - \mathbb H(X|Y)$


$\mathbb H(X,Y)$ is just the entropy of the joint variable $(X,Y)$. $\mathbb H(X|Y)$ is the expected entropy of $X$ if $Y$ is known. And $\mathbb I(X;Y)$ completes the elements of the ven diagram. It is the expected gain of information on $X$ (with gain of information I mean reduction of entropy) if $Y$ is known and vice versa, as mutual information is kommutative (see TODO insert venn diagramm). 


These terms act aditively as described by image TODO.


Please note that on Wikipedia equivalent but different definitions of these terms are stated. I choose these expressions as the definitions, as they are a) better to get an intuitive understaning and b) seem resemble more what the inventor had in mind, when coming up with these definitions. 


<!---
- briefly name an example applications from coding theory
- briefly name an example of neurology where neurons may maximize their entropy
-->



## The Link of Information Theory and LLMs


Information theory can be found everywhere, where uncertainty is expressed probabilisticly. As language models typically learn a probability distribution for the next token given a previous sequence parts maybe some new insights on them may be gained by looking at them from the viewpoint of information theory. How much information can a generative model generate, e.g. how much information is stored in this model? Can they solve tasks related to information theory such as data compression? 

In {{< cite "deletang2024language" >}} LLMs have been used as lossles data compressors and 



## Analyzing Neural Network Architectures {#anal_nn}


![targets](/figures/assumption_for_ideal_bayesian_estimator.png "Assumption AM") 


In this chapter results from {{< cite "jeon2022information" >}} and {{< cite "jeon2024information" >}} will be discussed. For several generative models models upper bounds have been established of how much information they can generate. These results can be used make statements about the expressiveness of different neuronal architecture. (<- be more precise about expressivemness)



# An Information Theoretic Perspective Analysis on In-Context Learning

In {{< cite "jeon2024information" >}} assumptions about the origin of the trianing data of LLMs have been made from which an explanation of in context learning was stated.




### What is In-Context Learning

In-context learning describes the phenomenon, where llms learned from the information inside the context window. (todo provide resouces). 
Examples on context learning are a, where b happened, c where d happend (todo provide examples from presentaitosn in class maybe?)


Here we also provide a brief example, where we tried to teach the anguage model ___ that war is good.

(TODO add screenshoooots)





### Results and Methodology {{< cite "jeon2024information" >}}
TODO: add theorem numerations of original paper.
#### Quick Overview


<!-- be consistent in the use of the word "bayesian prior" and "distribution of bayesian prior" -->

In [Analyzing Neural Network Architecture](#anal_nn) we discussed, how much information a given neural architecture can generate. The infered bounds can also be used to make statements about the training data probability space, if a certain assumption about that training data probability space was made. Which? The training data was generated by a random neural network of a given architecture. We will evaluate the pros and cons of such an assumption in [Discusion of their Assumptions](#discussion-assumptions). Once the distribution of the distribution (basically the bayesian prior) which generates the data  is formalized, we can make formal statements about the optimal (optimal with respect to a chosen loss function) estimator of the distribution of the training data. We call this estimator the optimal Bayesian estimator (OBE). If additionaly one assumes that a well trained transformer acts similarily as well as the oOBE, one can infer from the theoretic performance of the bayesian estimator on the performance on the transformer. When making these assumptions one has to be aware about certain dangers:
  - Is the bayesian prior distributio - the assumed distribution of distributions - plausible? Mor on that in [Discusion of their Assumptions](#discussion-assumptions)
  - The estiamtion of the OBE strongly depends on the bayesian prior distribution. If you change this distribution the OBE changes as well. If you assume transformers are really good, such that they are as good as an OBE, then the have to be as good as the OBE that makes correct assumptions on the bayesian prior

## The Assumed Bayesian Prior



![targets](/figures/assumption_sparse_mixture.png "Assumptio Sparse Mixture") 


Figure 1: The model of {{< cite "jeon2024information" >}} of the training data for LLMs. Each square represents a training Document, which has been randomly generated by a sparse mixture of AMs like transformers. Each Pink circle represents a randomly generated AM and for each document a random AM is assigned based on a random random distribution.  



In this paragraph we will formally state the probabilistic model of {{< cite "jeon2024information" >}} for the generation of the training data and in-context window of LLMs.

We denote the $M$ as the number of training documents and the training documents with $\{D_1,...D_M\}$. $D_i$ is the sequence of tokens in the i'th document. $H_{m,t} := (D_1,...,D_{m-1}, X_1^{(m)},...,X_t^{(m-1)}) $ is an abreviation for the sequence of tokens created by the tokens in the first $m-1$ documents and the first $t$ tokens in the $m$'th document. $D_{M+1}$ denotes in-context document.

We say the distributions of $\{D_1,...D_{M+1}\}$ can described by autoregressive models (as are transformers) that are parametrized by the random vectors $\{\theta_1,...,\theta_m \}$. We pack for practicality all parameter vectors into one $\theta := \{\theta_1,...,\theta_m \}$

The authors wanted to model $\theta_1,...,\theta_m $ such that they have some universal common information, which can be stored in a random variable $\psi$. This means that the sequence $\theta_1,...,\theta_m | \psi$ shall be iid. 

Additionally $\psi$ shall not contain information about any $\theta_m$ that could not be deducted from enough samples of $\theta_i$,   $D_m \bot \psi | \theta$. Since we sayd $\theta_1,...,\theta_m | \psi$ shall be iid, it holds  $D_m \bot \psi | \theta \iff D_m \bot \psi | \theta_m$. (<- not peer reviewed)

(<- Check if this is correct) 
How can the random sequence $\theta_1,...,\theta_m$ be modeled to satisfy that constraint in such a way, that the distributions of $\theta_1,...,\theta_m$ and $\psi$ are well enough defined. In {{< cite "jeon2024information" >}} the authors came up with the clever solution for satisfying these constraints. They defined a random set of $N$ randomly initialized autoregressive models $T = \{\theta^{(1)},..., \theta^{(N)} \}$, where $N$ is an unknown number. Then they defined a random assignment of Documents and autoregresive models in T parametrized by random random distribution $\alpha \sim \text{Dirichlet}(N, (R/N, ..., R/N))$, with $R<<N$. $\alpha$ defines for a random autoregressive model $\theta^{(n)}$ its probability to be assigned for Documents. For the case, in which the autoregressive models where transformers, $\theta^{(n)}$ represented a vector of gaussian independently distributed variables, describing the wheight parameters. The smaller the values in the parameter-tuple $(R/N,...,R/N)$ of the Dirichlet distribution, the more sparse the distribution. Lets suppose for example if you have a factory that produces factories, which in return produce each $k$ dices and you want the dieces to be fair (each side appears with same probability). Then the dirichlet distribution, parametrized by $k, (a,...,a)$, which describes the distribution of the probability distribution dices created by a given dice factory should have a high value of $a$ in its paramatrization. <!--example is unintuitive + maybe be a bit more formal what alpha does--> 
So now we can finally define $\psi := (\alpha, \theta^{(1)}, ..., \theta^{(n)} )$. Note that $\theta_m | \psi$ is a discrete random variable with at most $N$ outcomes, therefore its entropy has an upper bound of $\log N$.
Therefore if $M$ grows to infinity maybe the OBE will learn $\psi$ from $H_M^T$, e.g. $\log N \geq \mathbb H(\theta_{M+1} | \psi) \approx \mathbb H(\theta_{M+1}|H_M^T)$ and this may result in a logaritmic upper bound for the estimation error of $\theta_{M+1}|H_M^T$. You wonder what the estimation error is? This will be formaly definded in the next Paragraph. On top of that the previous claim will be formally evaluated in the next paragraph.


You might remember that earlier in this chapter I said that the authors assumed, that all training data has been generated by a transformer. And now I suddently presented a sparse mixture of transformers instead of a transformer. This is because in the conclusion the authors said, that they hope, how further mathematical analysis will be able to describe how a transformer can implement a sparce mixtrue of transformers. In Section "Can a transformer implement a sparse mixture of transformers?" (TODO add link finish proof), we show that a transformer can not approximate a sparse mixture of transformers. (But espilon appoximate if d grows to infinity?) 

So actually they made this assumption, but this assumption might be made in the future once it has been prooven that a transformer can implement a sparce mixtre of transformers.

(TODO proof that a sparse mixture of transformers with fiven transformers has an infinite horizon)

## Results for in-context learning

### Results without making assumptions about the bayesian prior 

In this paragraph we outline the results {{< cite "jeon2024information" >}} drew from this models of data generation, by analyzing the optimal bayesian estimator $\hat P$ for the probability distribution of $X^{(m)}_{t+1}$ given $H_t^{(m)}$.  

The optimal bayesian estimator is  defined to be  the estimator for the probability $P$ that minimizes this loss function:

$\mathbb L_{M,T}(P) = \frac{1}{TM} \sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln P(H_t^{(m)})(X_{t+1}^{(m)})]$

A little remark on the expression $P(H_t^{(m)})(X_{t+1}^{(m)})$: $P$ is a function that takes in an event like $H_t^{(m)}$, and returns a function, namely an estmated distribution for $X_{t+1}^{(m)}$. Therefore there are two braket pairs after $P$ in the above equation. 

In {{< cite "jeon2024information" >}} it was shown, that $\hat P(H_t^{(m)}) = (x \to \mathbb P[X_{t+1}^{(m)} = x|H_t^{(m)}])$ (If $X$ is not discrete the left equation has to be expressed slightly differently). 

Let's denote the loss of the minimal bayesian optimizer with $\mathbb L_{M,T} := \mathbb L_{M,T}(\hat P) = \frac{1}{TM} \sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]]$.

As autoregressive models such as transformers define a probability distribution one can not predict a sequence of tokens they generate with probability 1, even if one knows all of their parameters $\theta$ (Expect of course if they are deterministic autoregressive models).  This means that $\mathbb H(H_T^{(M)}|\theta)$, e.g. the hardness of predicting the sequence even if the autoregressive model is known, has a value above 0 and might grow to infinity with T or M. The Authors named this error "the irreducible error". As we are interessted in how hard it is to estimate not the series of tokens itself, but their distribution, the error $\mathcal L_{M,T} = \mathbb L_{M,T} - \dfrac{\mathbb H(H_T^{(M)}|\theta)}{MT}$ will be more insightfull in our exploration.
They call this error "estimation error".


In the rest of this section we will present derived expressions for $\mathcal L_{M,T}$, that provide insights in how easy or hard it is to estimate model parameters, if the data was generated as discussed. Then we derive from this result an expression of the in-context error $\mathbb L_\tau := \frac{1}{\tau} \sum_{t=0}^{\tau-1} \mathbb E \ln \mathbb P(X_{t+1}^{(M+1)}| H_t^{(M+1)})$, where $\tau$ is the length of the in-context document.

Firstly we discuss two information theoretic results of $\mathcal L_{M,T}$. 

----
#### Theorem Jeon.3.2 ({{< cite "jeon2024information" >}})

$\mathcal L_{M,T} = \dfrac{\mathbb I(H_T^{(M)};\theta)}{MT}$


(I adapted the original theorem slightly as the apaption fits better into this explanation.)

----
#### Proof:

We know $\mathbb H(X) = \mathbb H(X|Y) + \mathbb I(X;Y)$, therfore it suffices to show that $\mathbb L_{M,T} \cdot MT = \mathbb H(H_T^{(M)})$. This is true since:

$\sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]] = \mathbb E[-\ln \prod_{m=1}^{M} \prod_{t=0}^{T-1} \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]] \overset{a)}{=} \mathbb E[-\ln  \mathbb P[H_T^{(M)}]]= \mathbb H(H_T^{(M)})$


a) follows from the chain rule of probability.

QED

Disclaimer: This is my own version of the proof, I dont guarantee correctness.

----

As an intermediate result of the proof we obtained $\mathbb L_{M,T}  = \dfrac{\mathbb H(H_T^{(M)})}{MT}$. Thus we can see $\mathbb L_{M,T}$ as the average entropy per token.


This equation means roughly speaking, that the estimation error consits of these parts in $\theta$, that are conveyed to $H_T^{(M)}$. For example as we often model $\theta$ as a continous random variable and $H_T^{(M)}$ as a discrete random variable, $H_T^{(M)}$ can not contain all information in $\theta$. (<- be more clear)

As in previous section, we have worked out a way to separate $\theta = \theta_1, ...\theta_m$ into two independent random variables, namely $\theta|\psi$ and $\psi$, we continue by expressing $\mathcal L_{M,T}$ with these two random variables. 


----
#### Theorem Juan.4.2 ({{< cite "jeon2024information" >}})

$\mathcal L_{M,T} = \underbrace{\dfrac{\mathbb I(H_T^{(M)};\psi)}{MT}}_\text{meta
estimation error}$ + 

$+ \underbrace{\dfrac{\mathbb I(D_m;\theta_m|\psi)}{T}}_\text{intra document estimation error}$

(todo make  both  sum elements appear in the same line)

----

#### Proof

From Theorem Jeon.3.2 we know $\mathcal L_{M,T} = \dfrac{\mathbb I(H_T^{(M)};\theta)}{MT}$.

To proof this equation it suffices to show 

- a) $\mathbb I(H_T^{(M)};\theta) = \mathbb I(H_T^{(M)};\psi) + \mathbb I(H_T^{(M)};\theta|\psi) $
- b) $\mathbb I(H_T^{(M)};\theta|\psi) = M \cdot \mathbb I(D_m;\theta_m|\psi)$


We defined earlier $D_m \bot \psi | \theta_m$, which means $ H_T^{(M)} \bot \psi | \theta$ (<- be more formal?). Therefore $\mathbb I(H_T^{(M)};\theta) = \mathbb I(H_T^{(M)};(\theta, \psi))$. Then b) follows from the chain rule of mutual information.

b) follows from $\mathbb I(H_T^{(M)};\theta|\psi) \overset{b.1)}{=} \sum_{m=1}^M  \mathbb I(D_m;\theta_m|\psi) \overset{b.2)}{=} M\cdot \mathbb I(D_m;\theta_m|\psi) $. 

Equation b.1) holds because the pairs $(D_1, \theta_1)|\psi, ..., (D_M, \theta_M)|\psi$ are independent and mutual information is additive for independent variable pairs  {{< cite "latham2009" >}}.
As $ (D_1, \theta_1),..., (D_M, \theta_m) | \psi$ are identically distributed, for any $a,b \in \{1,...,M\}$, $\mathbb I(D_a;\theta_a|\psi) = \mathbb I(D_b;\theta_b|\psi)$. Therefore b.2) is true.



(<- proof not peer reviewed)
QED  


Disclaimer: This is my own version of the proof, I dont guarantee correctness.

---

The authors seperate the term into two parts, the meta estimation error and the intra document estimation error. The meta estimation error describes the error, of learning information shared shared by all documents. The intra document estimation error is the error of learning the parameters of learning individual documents after having learned the shared information $\psi$. Lets say we have arbitrarily many training samples $A$ for our optimal bayesian estimator in which $D_m$ does not occur. Due to them being arbitrarily many, let's suppose $\psi$ has been well enough discovered, such that $\mathbb I(D_m;\psi | A)$ is about 0. Then the error of estimating $D_m$ is about the intra-document error. This means, that even of we perfectly train our model, there will be some error bigger than zero related to the estimation of the probability that generates the data? Or will it realy be bigger than zero? More on this cliffhanger later (<- TODO!) 


### Results from assumptions of bayesian prior

In order to make the results about the error of the optimal bayesian estimator more concrete more assumptions about the bayesian prior are required. Unfortinately as discussed in section (TODO) our view on amount of information requires the distribution of the bayesian prior to be well enough known (in this case "well enough" usually means ""). Specifically for the work of {{< cite "jeon2024information" >}} a distribution of the training documents would be well-defined, if the AMs in the sparce mixture where well defined. Therefore as discussed in section (todo: add reference), the AMs where modeled as random transformers, where the transformer parameters $\theta^{(n)}$ are random variables that stem from the same distribution and are independent. 

Does the independence maximise the output entropy of the resulting tokensequence under the assumption, that the distribution of the actual training data sequence can be generated by a fixed parameter transfomer? If that was true, the upper bounds of {{< cite "jeon2024information" >}}, at which we will look in this section, would be an upper bound  for any distribution of the parameters of the transformers. In Section (Does independence suffice for an upper bound of estimation error?), I provide qualitative arguments that this is not the case.

Lets present their results.

Let 
  - $\Sigma$ be the set of tokens 
  - $d := |\Sigma|$ be the size of the vocabulary
  - $K$ be the context length 
  - $L$ be the transformer depth
  - $r$ be the attention dimension

To define the transformer architecture they define
  - $U_{t,l}$ to be the ouput of layer $l$ at time $t$
  - $U_{t,0}$ to be the embeding of the last $K$ tokens at time $t$
  - $\sigma$ to denote the softmax function
  - $A^{(l)} \in \mathbb R^{r \times r}$ to be the product of keay and query matrices of layer $l$
  - $V^{(l)} \in \mathbb R^{r \times r}$, $V^{(L)} \in \mathbb R^{d \times r}$ to denote the value matrix of layer $l$. 
  - $\text{Attn}^{(l)}(U_{t,l-1} = \sigma(\dfrac{U_{t,l-1}^T A^{(l)}U_{t,l-1}}{\sqrt r}))$ to be the attention matrix of layer $l$ 

Therefore $U_{t,l} = \text{Clip}(V^{(l)}U_{t,l-1}\text{Attn}^{(l)}(U_{t, l-1}))$

Given this transformer setup, they defined to be the parameters in the matrices $A_l$ to be distributed iid from $\mathcal N(0,1)$ and the parameters of the value matrices $V^{(l)}$ iid from $\mathcal N(0,1/r)$.


For the single sequence of tokens and single random transformer parametriued $\theta$ as described above they could show this opper bound for the estimation error $\mathcal L_T = \dfrac{\mathbb I(X_T;\theta)}{T}$:

-----

#### Therorem Juan.3.5 ({{< cite "jeon2024information" >}})

If $X_1,...,X_T$ is generated by the above defined transformer environment, then

$$\mathcal L_T \leq \dfrac{pL\ln(136 \text e K^2) + p \ln(\frac{2KT^2}{L})}{T}$$

, where $p = 2r^2(L-1) + (dr + r^2)$ denotes the parameter count of the transformer.

----

 This theorem does not only convey meaning for the case, in which $X_1, ..., X_T$ is generated as described above by a not well known (=random) transformer, but also how much "learned information" a trasformer can transmit per token to someone who doesnt know the parameters of the transformer.  This result could could be compared to the results of Juan et al 2022 for fully connected neural networks and other models, to get insight in which architectures can learn how many bits of information. In order to be properly compared the results of Juan et al 2022 have to be adapted a bit, to the same in and outputs. 
 (todo you might move above paragraph to section (Analyzing Neural Network Architectures)) The



For the case in which there are $M$ training documents and there is a sparse mixture of trasformers $\theta^{(1)}, ..., \theta^{(n)}$ another result for the upper bound of the estimation error $\mathcal L_{M,T} = \dfrac{\mathcal I(H_{M,T};\psi)}{MT} + \dfrac{\mathcal I(D_m;\theta_m|\psi)}{M}$.


#### Theorem Juan.4.5 ({{< cite "jeon2024information" >}})


If $D_1,...,D_M $ is generated by the above defined sparse mixture of transformers, then

$\mathcal L_T \leq 
\dfrac{pR\ln(1+\frac{M}{R} )\ln(136 \text{e} K^2) }{MT} + \dfrac{pR\ln(1+\frac{M}{R}) \ln(\frac{2KMT^2}{L})}{MT}+ \dfrac{\ln(N)}{T}$

, where $p = 2r^2(L-1) + (dr + r^2)$ denotes the parameter count of each transformer.


---------


(TODO talk about rate distortion error bounds?)


## Discusion of their Assumptions {#discussion-assumptions}

### Can we assume the existance of a transformer, generating all training documents

- transformers can imitate every upper bounded finite horizon am model
- but they cant imitate infinite horizon documents

----

#### Unformal Lemma 0: 

There exists at least one document $d$, which has a word in the very end $w_e$, which is statistically dependent on the very beginning $w_b$ of the document, even given the rest of the document. 

----

#### Unformal Proof: 
Scroll down to the very end of this document below the references. 

QED

---
### Can a transformer implement a sparse mixture of transformers?

!!IMPORTANTE IMPORTANTE!!: adapt notation to juan et al 2024!!!!!

(No a sparce mixture of transformers has an infinite horizon.)

In this section, we show that a sparce mixtrue of AMs can not be expressed by a set any finite horizon AM model. Furthermore we hypothesize which 

----
#### Definition: 
Let $A$ be a set and $n \in \mathbb N$, then **$A^{[n]} := \bigcup_{i\in \{0,1,...,n\}} A^i$**. 

----


$A^i$ can be seen as the set of all words of length $i$ over alphabet/tokenset $A$ and consequently $A^0$ as the set, that contains the empty word, given $A$ is not the empty set. 

----

#### TODO Define s_1, d_1

---

#### Definition: 
Two makrov-chains $M_1 = (P_1,V)$ and $M_2 = (P_2,V)$ are called **deterministically distinguishable (dd)** if there is transition $e \in V^2$, such that $P_1(e) + P_2(e)>0$ and $P_1(e) \cdot P_2(e) = 0$. (E.g. $M_1$ and $M_2$ are dd if one has a transition with probability zero, where the other has positive transition probability)

----

#### Lemma A: 
Let $M_1 = (P_1,V)$ and $M_2 = (P_2,V)$ be not dd but irreducible and finite makrov chains, $P_1 \neq P_2$ and $\theta \in (0,1)$. Let $x_0,X_1,X_2...$ be a random sequence in $V$, where $x_0$ is constant and the distribution of $x_0, X_1, ... $ follows with probability $\theta$, $M_1$ and with probability $1-\theta$, $M_2$. Then there is $t*$ such that for infinite $t>t*$ the finite horizon $d$ optimal bayesian estimator $\mathbb P(X_{t+1}|X_{t-d+1}, X_{t-d+2} ..., X_t)$ does not equal the unbounded optimal bayesian estimator $ \mathbb P(X_{t+1}|X_1, ..., X_t) $.

(TODO maybe you can replace irreduciblt with something weaker)
TODO make this : we assume M_theta = M_1 more formal somehow, TODO define more clearley what doe not equal 
#### Proof: 
We cdenote the random markov cahin, that is $M_1$ with probability $\theta$ and otherwise $M_2$, with $M_\theta$. We first show two statemetns:
- a) Probability of $M_\theta = M_1$ grows with t to one given sequenze H_t and if $M_1 = M_\theta$
- b) Probability of $M_\theta = M_1$ less than one, given t is less than one.

- a) $\lim_{t \to \infty} \mathbb P[M_\theta = M_1 | H_t] = 1$, if we assume $M_\theta = M_1$
- b)  For all $t \in \mathbb N$ it holds: $\mathbb P[M_\theta = M_1 | H_t] < 1$, if we assume $M_\theta = M_1$



##### Proof of a)  


Since $P_1 \neq P_2$ there must be $(x,y)\in V^2$, such that $P_1((x,y)) \neq P_2((x,y))$. Let $T_i$ denote the random variable that $M_\theta$ visits $x$ for the $i$'th time. As $M_1$ and $M_2$ are irreducible and finite for $i \in \mathbb N$, $T_i$ is finite with probability 1. (TODO proof, argument?) Note that $X_{T_i}(\omega) = x$. For convenience we denote $p_1 := P_1(x,y)$ and $p_2 := P_2(x,y)$.

Notice that the sequence $X_{T_1 +1}, X_{T_2 +1}, ... | M_\theta$ is iid. To make our life even more easier we define the binomial variable $Y_i = \bold 1[X_{T_i + 1} = y]$. Now $Y_1, Y_2... $ is an infinite sequence of coinflips, that land with probability $\theta$ with probability $p_1$ on heads and with probability $1-\theta$ with probability $p_2$ on heads.

Let $A_t = \bold 1 [|\dfrac{S_y}{t} - p_1| < |\dfrac{S_y}{t} - p_2|] $


$A_t \to_{a.s.} 1$ iif $M_1 = M_\theta$ (todo show)

proof of a) finished

##### Proof of b)

Broof by induction:
Assume $M_1 = M_\theta$, assume for $t \in \mathbb N$ it holds: $\mathbb P[M_\theta = M_1 | H_t] = p_t < 1$.

Then $\mathbb P[M_\theta = M_1 | H_t, X_{t+1}] = \dfrac{\mathbb P[M_\theta = M_1|H_t]}{\mathbb P[ X_{t+1}| H_t]}\mathbb P[X_{t+1} | M_\theta = M_1,H_t] $

$= \dfrac{p_t P_1(X_t, X_{t+1})}{p_tP_1(X_t, X_{t+1}) + (1-p_t)P_2(X_t, X_{t+1})} <1$


Proof of b) finished.

(todo make limit more formal, make this whole block below more formal!!)
As we know from a) $\lim_{t\to \infty} \mathbb P(X_{t+1}|H_t) = \lim_{t\to \infty} \mathbb P(X_{t+1}|H_t, M_\theta = M_1) = \lim_{t \to \infty} P(X_{t+1}|X_t, M_\theta = M_1) = P_1((X_{t+1},X_t))$. (<- be more formal)
But we also know from b) (TODO deal with non existance of d in b))that there is $0<p_t < 1$, s.t. $\lim_{t \to \infty} \mathbb P(X_{t+1}|X_{t-d+1},...,X_t ) = \lim_{t \to \infty} p_t \cdot P_1(X_t, X_{t+1}) + (1-p_t) \cdot P_2(X_t, X_{t+1})$. And as defined in the lemma there is x,y such that $P_1(x,y) \neq P_2(x,y)$ 

QED

---

Lemma A means the resulting probability distribution of $x_0, X_1, ... $ made in this simple construction can not be expressed by any finite horizon AM model, such as a transformer

----

#### Hypothesis B: 
when d grows to infinity posterior distribution can be epsilon approximated

Proof: Left out

----

#### Corolary of lemma A: 
An upper bounded context $d_1$ length Transformer $T$ can not express the probability distribution a non-deterministic mixture of two transformers $(R,J)$ (Romeo and Juliett).

#### Proof: 
We start by showing the possibility of a reduction: Any discrete autoregressive distribution over alphabet $\Sigma$ with horizon $d$, $f:\Sigma^{[d]} \to \mathcal P_{|\Sigma|}$ can be express by an equivalent Markov chain $M=(V_m,P_m)$, where $V_m = \Sigma^{[d]}$. With equivalent it is meant that the new random model, has the same irreducible error, and the same bayesian estimation error (todo be formal about what you mean with equivalent (equivalen with respekt to obe), explain meaning of f!)
To achieve the epuivalent markov chain, we construct $P_m$ like this: given $w \in \Sigma^{[d]}$, $x \in \Sigma$ and $f(w)(x) = p$, $P_m((w, s_1(w) \circ x)) = p$. If a transition $(a,b) \in V_m$ has not defined in the previous sentence (e.g. $s_1(a) \neq p_1(b)$), then its porbability is zero $P_m((a,b)) = 0$.

(TODO: adapt to new version of lemma A, e.g. chains are not dd and irreducible)
Therefore there exists an equivalent markov chain $M_R=(V,P_R)$  of $R$ and $M_J$ analogously of $J$. By lemma A the distribution of a non-deterministic mixture of $M_R$ and $M_J$ can not be approximated by a bounded optimal bayesian optimizer. But as the parameters of $T$ has context length $d$ it can only express distributions of auto regressive models with horizons of at most $d$. Thus $T$ can not implement the non-deterministic misxture of R and J.


QED 

----

#### Corolary of corolary of lemma A: 
An upper bounded context $d_1$ length Transformer $T$ can not express the probability distribution a non-deterministic mixture of n transformers $\{J_1,...,J_n\}$ for any n>2.

#### Proof (TODO)

Assume ad absurdum: asumme n>0 exists, st a mixture of n fhAms can be implemented by an fhAM.
TODO finish proof
  - reduction of determening of fhAM_a, fhAM_b to determening n fhAMs.
  - generate n-2 random fhAMs (fhAM_2, ..., fhAM_n), and generate sequence of tokens with p=0.5 by (fhAM_a, fhAM_b) and with 0.5 percent by (fhAM_2, ..., fhAM_n) 
  - this would mean after d tokens obe must know whether it is fhAM_a and fhAM_b or not
  - given that obe knows 

 (TODO in definition of non deterministic mixture) (Romeo and Juliett).

QED

----

#### Comments: 
We assumed in corolarry of Lemma A that R,J take every next token with positive probability. This assumption was necessary for the corrolary to be true in general. Imagine R would give for a word $w$ to be followed by a letter $x$ the probability zero and T not. Then the optimal estimator would disguise T as soon as x would follow after w. Does that mean it stays plausible that for a non deterministic mixture of tansfoermes, that are limited to only taking one of the k most likely letters as the next one,  to be implemented as one transformer. I believe only yes, if all parameters of R and J are perfektly well known. As soon there is some ambiguity in there parameters, suddenlty we can not be exactly sure wich one is among the top k next tokens and suddently again all tokens could appear with positve probability. But thats only a believe and I do not provide any qualitative argument on that.

Also if we assume that 


### Does independence suffice for an upper bound of estimation error?
<!---
Lets suppose there was a transformer, that could generate all training documents, that we have. (For this the documents should have an upper bound regarding length. Also the documents should be independent if the transformer is known). 

Now we want to find an upper bound of the estomation error of the obe, but we dont know the bayesian prior, we just
-->

Lets suppose our training ducuments had all an upper bound length T, such that there would be a transformer being able to express their distribution, as we assume that the class of transformers can imitate any finite horizon distribution. Lets also assume that for any of these distributions there is a transformer with upper and lower bounded parameters $[-a,a]$.

Now here comes the question: can we use this insight to find an upper bound for any optimal bayesian estimator assitiated to any prior bayesian assumption?

As transformers can express any distribution within our frame and if we start to sample them in the most random way (e.g. highest entropy), should we then get the most random distribution of distributions (e.g. the distribution of distribution, such that we can not find a baesian prior, such that the obe achieves a higher error)? Can we find, what the unifotm distribution is for finite discrete spaces, for AG probability distributions?

If we would find that upper bound, then we would have found an upper bound, which holds independent of the bayesian prior and we would never have to worry again if your assumptions are valid!!! So here is why it fails:


As the parameter space is bounded $[a,a]$, the highes entropy would be from $\theta ~ unif^n$ e.g. all thetas iid. Lets call this $\Theta$

Intuitively if using $A_t(\Theta)$, should work to create the highest bayesian error, then for any $A_x(\Theta)$ this should work as well as there is nothing special with transfomers. $A_x$ could be like $A(\Theta)$ but first convert all thetas to gaussians. 


Probabilitstik correct answer:
since we have a finite horizon T and a finite alphabet $\Sigma$, we have a finitely large space of outcomes. Let $\mathbb X$ be such a random token.
Thus the problem can be reduced to estimating a bernoulli dsitribution.
(Is it estimating a berounlli distribution, or estimating a distribution of bernoulli distriution)

The estimation error comes down to

$\mathbb I[\mathbb X; \Theta]$ (upper bound log(d)T)


argmax distribution of theta $\mathbb I[\mathbb X; \Theta] = \mathbb H[\mathbb X] - \mathbb H[\mathbb X | \Theta]$

<!-- see the whole context space as one character -->

Be aware that H(X) is not indepenent on the distribution of theta.

explain why problem is ultra hard to solve, explain on easy example. 


X itself is the hardest to estimate if it is uniform distributed. But if we know that X is uniform distributed, then estimation error is minimal. 
if estimation error is bigger than zero, interesstingly X is not for sure uniform distrivuted (if we kew for sure X was uniform distributed then estimation error was zero, as we would then have no uncertainty about distribution)). Does this observation challenge the meaning of the estimation error? Maybe we should not go about maximizing the estiamtion error?


### What is the meaning of the estimation error


In Section (TODO add reference), we cited this result for the error of the obe:

$\mathbb L_{M,T} MT= \mathbb H[H_M^{(T)}] = \mathbb I(H_M^{(T)};\theta) + \mathbb H(H_M^{(T)}|\theta)$

And then the estimation error $\mathbb I(H_M^{(T)};\theta)$ got extra care in the further analysis, as it conveys how hard it is to guess model parameters.


As we will argue in this Section, the estimation error is fully dependent on the bayesian prior model and independent of the resulting distribution. 
E.g. a sequence generated by a random transformer, can also be generated with the same distribution by a high amount of other random or non random models.

----
#### Lemma B.1

Every autoregressive distribution $X_1, ..., X_T$ over the finite alphabet $\Sigma$ can be represented by a parametrized model with an estimation error of 0.

(use functions A and K, use uniform instead of gausian bayesian prior)


#### Proof
Let $\mathcal P_{\Sigma}$ the set of all probability distributions over $\Sigma$.
Let $\mathcal P_{T} = \\{ \Sigma^{[T-1]} \to P_\Sigma \\}$ the set of all autoregressive probability distributions with $T$ tokens. 
An element in $\mathcal P_{T}$ can be seen to take in a sequnce of tokens and to return the probability distribution for the next token.

We call $A: \mathbb R^n \to P_T$ an autoregressive model. If a model $A$ is surjective, we call $A$ a full autoregressive model, since it can parametrize all autoregressive distributions. 

At least one full model exists, since $\mathbb R^n$ and $\mathcal P_T$ have the same cardinality.

A transformer is for example an autoregressive model. I dont know whether a transformers with context length T can $\epsilon$-approximate a full autoregressive model.



Let $H_T$ be representable by a full autoregressive model, such that its estimation error is bigger than zero. This means there is a a full autoregressive model $A$
and $\mathcal Y \in \mathcal P_{\mathbb R^n}$ and $\theta \sim \mathcal Y$ ($\theta$ be a random vector of some distribution), such that $H_T \sim A(\theta)$ and $\mathbb I(H_T;\theta)>0$.

Notice that $A(\theta) \in \mathcal P_T$. As $A$ is surejectiv there is constant $\theta_0 \in \mathbb R^n$, such that $A(\theta_0) = A(\theta)$ and thus $H_T \sim A(\theta_0)$.

If we model $H_T \sim A(\theta_0)$, then $\mathbb H(\theta_0) = 0$ and therefore we obtain an estimation error of $0$: $\mathbb I(H_T;\theta_0) = 0$.

(TODO move definition of full autoregressive model above lemma statement maybe into previous section)

QED

----


In Lemma B.1 we found, that we can do such a shitty job when modeling a dstribution, that by looking at the data and updating our model parameters accordingly we will never be able to describe a autoregressive distribution more precise, the more tokens we observe, e.g. we will never have to update them. (Still the model would become better at predicting the distribution by observing it, but that information would not be stored in updated model parameters.)

(todo define parametrization)
Can we also do the oposite? Imagine for a given random sequence $H_T$ we can find a parametrization $A, \mathcal Y$, s.t. the estimation error is maximal. Since this parametrization would maximize the estimation error, it would minimize $H(H_T|\Theta)$.
Would we thereby find a model, such that it parameters can describe the sequence the best? Would we therby find the "perfect" model for this sequence? 
Lemma B.2 shows, that for every sequence there exists a parametrization, such that $H(H_T|\Theta) = 0$, but when you read the proof you will be disappointed.



- write how cool it would be if we where able to find for a given X distribution, the bayesian prior that maximises the estimation error
  - would the resulting model then be the perfect model for this distribution?
    - this is the model that would minizie H(X|Theta)   
    - this means that for many samples better estimation performance? (no bayesian prior always perfekt, but we are not, maximal possible estimation error tells us how much randomness will be removed for many samples) 
  - for example uniform distribution forces estimation error to be zero, e.g. every modelling of uniform distribution is equal as shit


----

#### Lemma B.2

Every autoregressive distribution $H_T=X_1, ..., X_T$ can be represented by a model with an estimation error of $\mathbb H(H_T)$.


#### Proof

(doenst have to be a bijection)

Let $\Sigma$ be the alphabet of $H_T$. Let $f: \Sigma^\infty \to \mathbb R^n$ be a bijection.

We define $A(\theta) := h \to (x \to \mathbb P[f^{-1} (\theta)_{|h|+1}  = x)| f^{-1}(\theta)_{1:|h|} = h]$ 
 where $f^{-1}(\theta)_{i:j}$ defines the word that goes from $i$'th cahracter of the word $f^{-1}(\theta)$ until the $j$'th character.
I am sorry that this expression looks confusing but I will try to explain: $f^{-1}(\theta)$ is just an invinite long word. $A(\theta)(h)(x)$ returns 1 iif the word $h \circ x$ is a prefix of $f^{-1}(\theta)$. If $h$ is a prefix of $f(\theta)$, but $h \circ x$, then $A(\theta)(h)(x) = 0$ and otherwise $A(\theta)$ is undefined. 

Note that $A$ is a not well-defined autoregressive model, but by far not a full autoregressive model.

Now for a given autoregressive sequence $H_T$, we define the random vector $\theta = f(H_T \circ 0^\infty)$.

Then $A(\theta)(h_t)(x) = \mathbb P[f^{-1} (\theta)_{t+1}  = x| f^{-1}(\theta)_{1:t} = h_t] = \mathbb P[X_{t+1}  = x| H_t = h_t]$, e.g. $H_T \sim A(\theta)$.




----


- write how cool it would be if we where able to find for a given X distribution, the bayesian prior that maximises the estimation error
  - would the resulting model then be the perfect model for this distribution?
    - this is the model that would minizie H(X|Theta)   
    - this means that for many samples better estimation performance? (no bayesian prior always perfekt, but we are not, maximal possible estimation error tells us how much randomness will be removed for many samples) 
  - for example uniform distribution forces estimation error to be zero, e.g. every modelling of uniform distribution is equal as shit



## Conclusion

In this blog post several applications of Information Theory in Deep Learning and LLMs where presented. 

## Take home messages


# References
{{< bibliography >}} 


https://perchance.org/emoji



The emoji-character sequence from the beginning of the blogpost was generated randomly, after the text of this Blog Post has been written, it is: ⚠️ 📥 😚 🛡 🚦 👹 🌿 