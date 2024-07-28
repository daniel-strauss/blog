---
title: Some Information Theoretic Perspectives on LLM Theory
bibFile: data/bibliography.json # path relative to project root
---
Author: Daniel Strauß, Supervisor: Suvrit Sra

⚠️ 📥 😚 🛡 🚦 👹 🌿 *By reading this blogpost, you will find out, why this emoji sequence is here.*


TODO rewrite thetas, random theta: big, constant theta small

TODO: when you reference them, say "they", when you reference you say "we"

TODO correct all sentences with Grammarly

## Introduction

In this blog post, you will first learn about information theory, and we will look at one example of the application of information theory in LLM theory. Then, we will look at one paper in detail that specifically examines the phenomenon of in-context learning by making assumptions about the probability distribution of the training data {{< cite "jeon2024information" >}}. Next, we will carefully examine the assumptions made in that paper. To do that well, we will prove some of our own statements based on the results of that paper.

- The authors called for further analysis of how a transformer can implement a sparse mixture of transformers. In this blog post, we prove that a transformer cannot implement a sparse mixture of transformers.
- Furthermore, we prove that any discrete autoregressive distribution can be modeled both such that the estimation error defined in that paper is zero and that the estimation error equals the entropy of the distribution. We discuss this result and state our opinion on how to use this result to better interpret the estimation error.

### A Brief Introduction to Information Theory

![targets](/figures/venn_information.png "Assumptio Sparse Mixture") 

 

Imagine you lived 40⋅n40\cdot n years in the future and the population has doubled every 40 years. For simplicity, assume that in the current present, just one person is alive. Conclusively 2n2^n  people live at your point in the future. You live very far in the future and nn is very large $n=2^{10^{15}}$. "Today" is your friends birthday party and you dont have a present. 

Therefore you decide to hire last minute a mathgician for his party as mathagicians are know to let partys go wild with their magic and math. 
In order to hire the mathgician you have to put in your friends one petabit (=1015=10^{15} bits) long address into the website of the mathgician-firm, such that he can find the location of the party. (The address is one petabit long as in the future every inhabitant has his own address, and there are nn people.) At the party the mathgician shows up and everyone has a good time. But then suddenly you start to wonder:

You only have a 1Gbit/s upload speed (hardware did not improve so much in the last $40\cdot2^{10^{15}}years),howhaveyoubeenabletoupload years), how have you been able to upload 10^{15}bits(onePentaBit)withinjustoneday?Thisseemsimpossibleasittakes bits (one PentaBit) within just one day? This seems impossible as it takes 10^6secondstouploadonePentaBitwithyournetworkspeed.Thenyoufindoutwhy:thewebsitedidnothavetosend seconds to upload one PentaBit with your network speed. Then you find out why: the website did not have to send 10^{15}$ bits of information to its server. Small parts of the information have already been known by the server; For example the information that people in your region are more likely to order a magician for someone in the same galaxy was already present on their server. 
Could they have used that information to such that they required less bits of information from you, such that your computer had to send them a smaller amount of bits? How can something seeming as soft as information influence something as hard as a bit-sequence length? What is the math behind this?


Information theory is mainly the study on expressing the quantity of information called Entropy. Entropy of an information is the minimal expected amount of a quantity like bits required to encode this information.

One can show, that the entropy H\mathbb H in bits of a discrete random variable XX with a known probability density function pXp_X is

$\mathbb H(X) = \sum_{x \in \mathcal X} - p_X(x) \log_2(p_X(x)) = \mathbb E[-\log_2 X]$. {{<cite "shannon1948mathematical">}}

Therefore the above term is usually referred as the formal definition of entropy. Often instead of the binary logarithm the natural logarithm is used to express entropy. The resulting unit is called nats instead of bits. 

H(X)=∑x∈X−pX(x)log(pX(x))\mathbb H(X) = \sum_{x \in \mathcal X} - p_X(x) \log(p_X(x))

The entropy of continoous random variables is infinity, as they can have infinetly many outcomes (TODO provide source). But the differential entropy \boldh\bold h of a continous random variable. (TODO explain use of diferential entropy)


$\bold h(X) = \mathbb E[-\log(X)]$


Information theory provides several expressions for how the information of different random variables relate.  This is useful in many different scenarios. (<- TODO example)
Definition:

- Joint Entropy: H(X,Y):=H((X,Y))\mathbb H(X,Y) := \mathbb H((X,Y))
- Conditional Entropy: $\mathbb H(X|Y) := \mathbb E[(y \to \mathbb H(X|Y=y))(Y)] = \sum_{y \in \mathcal Y} p_y(y) \sum_{x \in \mathcal X} p_{X|Y=y}(x) - \log(p_{X|Y=y}(x))$ (<- todo chec correct)
- Mutual Information: I(X;Y):=H(X)−H(X|Y)\mathbb I(X;Y) := \mathbb H(X) - \mathbb H(X|Y)


H(X,Y)\mathbb H(X,Y) is just the entropy of the joint variable (X,Y)(X,Y). H(X|Y)\mathbb H(X|Y) is the expected entropy of XX if YY is known. And I(X;Y)\mathbb I(X;Y) completes the elements of the ven diagram. It is the expected gain of information on XX (with gain of information I mean reduction of entropy) if YY is known and vice versa, as mutual information is kommutative (see TODO insert venn diagramm). 


These terms act aditively as described by image TODO.


Please note that on Wikipedia equivalent but different definitions of these terms are stated. I choose these expressions as the definitions, as they are a) better to get an intuitive understaning and b) seem resemble more what the inventor had in mind, when coming up with these definitions. 


<!---
- briefly name an example applications from coding theory
- briefly name an example of neurology where neurons may maximize their entropy
-->



### Data Compression and LLMs

As one might be able deduct from the definition of entropy made in previous Section, there is a strong link between data compression and information theory. Entropy of an information source is the amount of bits its optimal lossless compression holds on average. Basically ever since humanity developed new lossles compression techniques to express the same data with less bits. To do that, modeling the probability space, that generates the data, well is helpfull. In several work neural networks as transformers or other architectures have been used to compress data and ahieved good performances. <!--- in the online (Bellard, 2021; Mao et al., 2022) and offline settings (Valmeekam et al., 2023).--> As data compresson and data generation usually both rely on having some kind of representation of the datas probability distribution data generators, seem to be well suited for data compression.


Also in {{< cite "deletang2024language" >}} the performance of several LLMs as lossles data compressors has been evaluated. They did not only test the performance of LLMs on the compression of text data, but also on image and audio data. Interestingly the compresion rates of LLMs, such as of Llama 2 (7B) and of Chinchilla 70B, outperformed PNG on the compression on image data, and FLAC on the compression on audio data. 
Note, that if you add the size of the LLMs themleves into the compression rate, then a high amount of data has to be compressed, before a positive compression rate has been reached.

Whilst using these LLMs to compress data, has the big disatvantage that the LLMs themselves take up a lot of space and that compression with an LLM is computationally expensive, I still am impressed by that result. Since the LLMs in that work where primarily trained on text-data, I am surprised that they showed such good results in the compression of image and audio data. This shows that the probability spaces generating text vs image or audio data have common features. There seems to exist a nature of the data we encounter. On the other hand a non-neglectable fraction of that common nature might be not hard to find (not hard = humanly understandable). By examining humanly understanable compression algorithms, such as LZMA2, that also perform well on all three types of data one might be able to learn a good part of these commonalities, but not all, since LLMs still performed a bit better {{< cite "deletang2024language" >}}. 



<!---
## Analyzing Neural Network Architectures using Information Theory {#anal_nn}
-->

## An Information Theoretic Perspective Analysis on In-Context Learning

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

We denote the MM as the number of training documents and the training documents with {D1,...DM}\lbrace D_1,...D_M\rbrace. DiD_i is the sequence of tokens in the i'th document. $H_{m,t} := (D_1,...,D_{m-1}, X_1^{(m)},...,X_t^{(m-1)}) isanabreviationforthesequenceoftokenscreatedbythetokensinthefirst is an abreviation for the sequence of tokens created by the tokens in the first m-1documentsandthefirst documents and the first ttokensinthe tokens in the m′thdocument.'th document. D_{M+1}$ denotes in-context document.

We say the distributions of {D1,...DM+1}\lbrace D_1,...D_{M+1}\rbrace can described by autoregressive models (as are transformers) that are parametrized by the random vectors {θ1,...,θm}\lbrace \theta_1,...,\theta_m \rbrace. We pack for practicality all parameter vectors into one θ:={θ1,...,θm}\theta := \lbrace \theta_1,...,\theta_m \rbrace

The authors wanted to model θ1,...,θm\theta_1,...,\theta_m  such that they have some universal common information, which can be stored in a random variable ψ\psi. This means that the sequence θ1,...,θm|ψ\theta_1,...,\theta_m | \psi shall be iid. 

Additionally ψ\psi shall not contain information about any θm\theta_m that could not be deducted from enough samples of θi\theta_i,   Dm⊥ψ|θD_m \bot \psi | \theta. Since we sayd θ1,...,θm|ψ\theta_1,...,\theta_m | \psi shall be iid, it holds  Dm⊥ψ|θ⟺Dm⊥ψ|θmD_m \bot \psi | \theta \iff D_m \bot \psi | \theta_m. (<- not peer reviewed)

(<- Check if this is correct) 
How can the random sequence θ1,...,θm\theta_1,...,\theta_m be modeled to satisfy that constraint in such a way, that the distributions of θ1,...,θm\theta_1,...,\theta_m and ψ\psi are well enough defined. In {{< cite "jeon2024information" >}} the authors came up with the clever solution for satisfying these constraints. They defined a random set of NN randomly initialized autoregressive models T={θ(1),...,θ(N)}T = \lbrace \theta^{(1)},..., \theta^{(N)} \rbrace, where NN is an unknown number. Then they defined a random assignment of Documents and autoregresive models in T parametrized by random random distribution α∼Dirichlet(N,(R/N,...,R/N))\alpha \sim \text{Dirichlet}(N, (R/N, ..., R/N)), with R<<NR<<N. α\alpha defines for a random autoregressive model θ(n)\theta^{(n)} its probability to be assigned for Documents. For the case, in which the autoregressive models where transformers, θ(n)\theta^{(n)} represented a vector of gaussian independently distributed variables, describing the wheight parameters. The smaller the values in the parameter-tuple (R/N,...,R/N)(R/N,...,R/N) of the Dirichlet distribution, the more sparse the distribution. 
So now we can finally define ψ:=(α,θ(1),...,θ(n))\psi := (\alpha, \theta^{(1)}, ..., \theta^{(n)} ). Note that θm|ψ\theta_m | \psi is a discrete random variable with at most NN outcomes, therefore its entropy has an upper bound of logN\log N.
Therefore if MM grows to infinity maybe the OBE will learn ψ\psi from HTMH_M^T, e.g. logN≥H(θM+1|ψ)≈H(θM+1|HTM)\log N \geq \mathbb H(\theta_{M+1} | \psi) \approx \mathbb H(\theta_{M+1}|H_M^T) and this may result in a logaritmic upper bound for the estimation error of θM+1|HTM\theta_{M+1}|H_M^T. You wonder what the estimation error is? This will be formaly definded in the next Paragraph. On top of that the previous claim will be formally evaluated in the next paragraph.


You might remember that earlier in this chapter I said that the authors assumed, that all training data has been generated by a transformer. And now I suddently presented a sparse mixture of transformers instead of a transformer. This is because in the conclusion the authors said, that they hope, how further mathematical analysis will be able to describe how a transformer can implement a sparce mixtrue of transformers. In Section "Can a transformer implement a sparse mixture of transformers?" (TODO add link finish proof), we show that a transformer can not approximate a sparse mixture of transformers. (But espilon appoximate if d grows to infinity?) 

So actually they made this assumption, but this assumption might be made in the future once it has been prooven that a transformer can implement a sparce mixtre of transformers.


## Results for in-context learning

### Results without making assumptions about the bayesian prior 

In this paragraph we outline the results {{< cite "jeon2024information" >}} drew from this models of data generation, by analyzing the optimal bayesian estimator ˆP\hat P for the probability distribution of X(m)t+1X^{(m)}_{t+1} given H(m)tH_t^{(m)}.  

The optimal bayesian estimator is  defined to be  the estimator for the probability PP that minimizes this loss function:

$\mathbb L_{M,T}(P) = \frac{1}{TM} \sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln P(H_t^{(m)})(X_{t+1}^{(m)})]$

A little remark on the expression $P(H_t^{(m)})(X_{t+1}^{(m)}):: Pisafunctionthattakesinaneventlike is a function that takes in an event like H_t^{(m)},andreturnsafunction,namelyanestmateddistributionfor, and returns a function, namely an estmated distribution for X_{t+1}^{(m)}.Thereforetherearetwobraketpairsafter. Therefore there are two braket pairs after P$ in the above equation. 

In {{< cite "jeon2024information" >}} it was shown, that $\hat P(H_t^{(m)}) = (x \to \mathbb P[X_{t+1}^{(m)} = x|H_t^{(m)}])(If (If X$ is not discrete the left equation has to be expressed slightly differently). 

Let's denote the loss of the minimal bayesian optimizer with $\mathbb L_{M,T} := \mathbb L_{M,T}(\hat P) = \frac{1}{TM} \sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]]$.

As autoregressive models such as transformers define a probability distribution one can not predict a sequence of tokens they generate with probability 1, even if one knows all of their parameters θ\theta (Expect of course if they are deterministic autoregressive models).  This means that H(H(M)T|θ)\mathbb H(H_T^{(M)}|\theta), e.g. the hardness of predicting the sequence even if the autoregressive model is known, has a value above 0 and might grow to infinity with T or M. The Authors named this error "the irreducible error". As we are interessted in how hard it is to estimate not the series of tokens itself, but their distribution, the error LM,T=LM,T−H(H(M)T|θ)MT\mathcal L_{M,T} = \mathbb L_{M,T} - \dfrac{\mathbb H(H_T^{(M)}|\theta)}{MT} will be more insightfull in our exploration.
They call this error "estimation error".


In the rest of this section we will present derived expressions for LM,T\mathcal L_{M,T}, that provide insights in how easy or hard it is to estimate model parameters, if the data was generated as discussed. Then we derive from this result an expression of the in-context error Lτ:=1τ∑τ−1t=0ElnP(X(M+1)t+1|H(M+1)t)\mathbb L_\tau := \frac{1}{\tau} \sum_{t=0}^{\tau-1} \mathbb E \ln \mathbb P(X_{t+1}^{(M+1)}| H_t^{(M+1)}), where τ\tau is the length of the in-context document.

Firstly we discuss two information theoretic results of LM,T\mathcal L_{M,T}. 

----
#### Theorem Jeon.3.2 ({{< cite "jeon2024information" >}})

LM,T=I(H(M)T;θ)MT\mathcal L_{M,T} = \dfrac{\mathbb I(H_T^{(M)};\theta)}{MT}


(I adapted the original theorem very slightly as the apaption fits better into this post.)

----
#### Proof:

We know H(X)=H(X|Y)+I(X;Y)\mathbb H(X) = \mathbb H(X|Y) + \mathbb I(X;Y), therfore it suffices to show that LM,T⋅MT=H(H(M)T)\mathbb L_{M,T} \cdot MT = \mathbb H(H_T^{(M)}). This is true since:

$\sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]] = \mathbb E[-\ln \prod_{m=1}^{M} \prod_{t=0}^{T-1} \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]] \overset{a)}{=} \mathbb E[-\ln  \mathbb P[H_T^{(M)}]]= \mathbb H(H_T^{(M)})$


a) follows from the chain rule of probability.

QED

Disclaimer: This is my own version of the proof, I dont guarantee correctness.

----

As an intermediate result of the proof we obtained LM,T=H(H(M)T)MT\mathbb L_{M,T}  = \dfrac{\mathbb H(H_T^{(M)})}{MT}. Thus we can see LM,T\mathbb L_{M,T} as the average entropy per token.


This equation means roughly speaking, that the estimation error consits of these parts in θ\theta, that are conveyed to H(M)TH_T^{(M)}. For example as we often model θ\theta as a continous random variable and H(M)TH_T^{(M)} as a discrete random variable, H(M)TH_T^{(M)} can not contain all information in θ\theta. (<- be more clear)

As in previous section, we have worked out a way to separate θ=θ1,...θm\theta = \theta_1, ...\theta_m into two independent random variables, namely θ|ψ\theta|\psi and ψ\psi, we continue by expressing LM,T\mathcal L_{M,T} with these two random variables. 


----
#### Theorem Jeon.4.2 ({{< cite "jeon2024information" >}})

$\mathcal L\_{M,T} = \underbrace{\dfrac{\mathbb I(H\_T^{(M)};\psi)}{MT}}\_\text{meta
estimation error} + \underbrace{\dfrac{\mathbb I(D_m;\theta_m|\psi)}{T}}\_\text{intra document estimation error}$



----

#### Proof

From Theorem Jeon.3.2 we know $\mathcal L_{M,T} = \dfrac{\mathbb I(H_T^{(M)};\theta)}{MT}$.

To proof this equation it suffices to show 

- a) $\mathbb I(H_T^{(M)};\theta) = \mathbb I(H_T^{(M)};\psi) + \mathbb I(H_T^{(M)};\theta|\psi) $
- b) $\mathbb I(H_T^{(M)};\theta|\psi) = M \cdot \mathbb I(D_m;\theta_m|\psi)$


We defined earlier $D_m \bot \psi | \theta_m$, which means $ H_T^{(M)} \bot \psi | \theta$ (<- be more formal?). Therefore $\mathbb I(H_T^{(M)};\theta) = \mathbb I(H_T^{(M)};(\theta, \psi))$. Then b) follows from the chain rule of mutual information.

b) follows from $\mathbb I(H_T^{(M)};\theta|\psi) \overset{b.1)}{=} \sum_{m=1}^M  \mathbb I(D_m;\theta_m|\psi) \overset{b.2)}{=} M\cdot \mathbb I(D_m;\theta_m|\psi) $. 

Equation b.1) holds because the pairs $(D_1, \theta_1)|\psi, ..., (D_M, \theta_M)|\psi$ are independent and mutual information is additive for independent variable pairs  {{< cite "latham2009" >}}.
As $ (D_1, \theta_1),..., (D_M, \theta_m) | \psi$ are identically distributed, for any $a,b \in \lbrace 1,...,M\rbrace$, $\mathbb I(D_a;\theta_a|\psi) = \mathbb I(D_b;\theta_b|\psi)$. Therefore b.2) is true.




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

#### Therorem Jeon.3.5 ({{< cite "jeon2024information" >}})

If $X_1,...,X_T$ is generated by the above defined transformer environment, then

$$\mathcal L_T \leq \dfrac{pL\ln(136 \text e K^2) + p \ln(\frac{2KT^2}{L})}{T}$$

, where $p = 2r^2(L-1) + (dr + r^2)$ denotes the parameter count of the transformer.

----

 This theorem does not only convey meaning for the case, in which $X_1, ..., X_T$ is generated as described above by a not well known (=random) transformer, but also how much "learned information" a trasformer can transmit per token to someone who doesnt know the parameters of the transformer.  This result could could be compared to the results of Jeon et al 2022 (todo add resouce) for fully connected neural networks and other models, to get insight in which architectures can learn how many bits of information. In order to be properly compared the results of Jeon et al 2022 have to be adapted a bit, to the same in and outputs. 
 (todo you might move above paragraph to section (Analyzing Neural Network Architectures)) The



For the case in which there are $M$ training documents and there is a sparse mixture of trasformers $\theta^{(1)}, ..., \theta^{(n)}$ another result for the upper bound of the estimation error $\mathbb L_{M,T} = \dfrac{\mathcal I(H_{M,T};\psi)}{MT} + \dfrac{\mathbb I(D_m;\theta_m|\psi)}{T}$.


#### Theorem Jeon.4.5 ({{< cite "jeon2024information" >}})


If $D_1,...,D_M $ is generated by the above defined sparse mixture of transformers, then

$\mathcal L_T \leq 
\dfrac{pR\ln(1+\frac{M}{R} )\ln(136 \text{e} K^2) }{MT} + \dfrac{pR\ln(1+\frac{M}{R}) \ln(\frac{2KMT^2}{L})}{MT}+ \dfrac{\ln(N)}{T}$

, where $p = 2r^2(L-1) + (dr + r^2)$ denotes the parameter count of each transformer.


---------


As said in {{< cite "jeon2024information" >}} the first two terms relate to learning $\psi$, e.g. to $\dfrac{\mathbb I(H_{M,T};\psi)}{MT}$.
The third part relates to learning, which transformer of the mixture was generating the document, e.g. $\dfrac{\mathbb I(D_m;\theta_m|\psi)}{T}$. As we can see the first two parts converge to zero with a large number of training documents $M$. The upper bount is indebendent of $N$ due to the ditrichilet assumption; As $N$ grows, the ditrichilet parameter $R/N$ becomes smaler and the mixture more sparse. Regarding the third part, I think it might be not too difficult to find a lower upper bound. As $\theta_m|\psi \sim \alpha$, calculating the average entropy of $\theta_m|\psi$ as an upper bound for the in-context error might be not to impossible and due to the sparcity I am sure that it will be lower than $\ln N$.



What do these results tell us about in-context learning? The in-context error $\mathbb L_\tau$ is defined to be the error of the optimal bayesian estimator of predicting the in-context Document $D_{M+1}$ after having observed all other documents, where $\tau<K$ is the lenght of the in-context document.

$\mathbb L_\tau := \frac{1}{\tau} = \sum_{t=0}^{\tau-1} \mathbb E[- log P[X^{(M+1)}_{t+1}| D_t^{M+1} ]$

Then we denote the irreducible error as $\mathcal L_\tau := \mathbb L_\tau - \dfrac{\mathbb H[D_{M+1}|\theta^{(M+1)}]}{\tau}$.

----
#### Theorem Jeon.4.7

$\mathcal L_\tau \leq \dfrac{\mathbb I[H_{M,T};\psi}{M\tau} + \dfrac{\mathbb I(D_{M+1}; \theta_{M+1}|\tau)}{\tau}$.

----

From Theorem Jeon.4.5 we know that the first term will converge to zero with large amount of training data $M$ for the saprce mixtrue of transformers assumption.
For this assumption the right-hand term will be upper bounded by $\log(N)/\tau$, as only the model from the sparce mixtrue has to be distinguished. If $\log(N)/\tau$ is a low number of nats, then the sparse mixture assumption can explain, how in-context learning works well. In the next chapter we will discuss these assumptions. 


## Discusion of their Assumptions {#discussion-assumptions}

### Can we assume the existance of a transformer, generating all training documents


----

#### Unformal Lemma 0: 

There exists at least one document $d$, which has a word in the very end $w_e$, which is statistically dependent on the very beginning $w_b$ of the document, even given the rest of the document. 

----

#### Unformal Proof: 
Scroll down to the very end of this document below the references. 

QED

---
### Can a transformer implement a sparse mixture of transformers?

!!IMPORTANTE IMPORTANTE!!: adapt notation to Jeon et al 2024!!!!!

(No a sparce mixture of transformers has an infinite horizon.)

In this section, we show that a sparce mixtrue of AMs can not be expressed by a set any finite horizon AM model. Furthermore we hypothesize which 

----
#### Definition: 
Let $A$ be a set and $n \in \mathbb N$, then **$A^{[n]} := \bigcup_{i\in \lbrace 0,1,...,n \rbrace} A^i$**. 

----


$A^i$ can be seen as the set of all words of length $i$ over alphabet/tokenset $A$ and consequently $A^0$ as the set, that contains the empty word, given $A$ is not the empty set. 

----

#### TODO Define s_1, d_1

---

#### Definition: 
Two makrov-chains $M_1 = (P_1,V)$ and $M_2 = (P_2,V)$ are called **deterministically distinguishable (dd)** if there is transition $e \in V^2$, such that $P_1(e) + P_2(e)>0$ and $P_1(e) \cdot P_2(e) = 0$. (E.g. $M_1$ and $M_2$ are dd if one has a transition with probability zero, where the other has positive transition probability)

----

In lemma A we show, that in order to express the probability space of a non-deterministic mixture of two makrov chains, one needs an unbounded knotext length. This means there can not be a finite knotext length model that implements the probability space a non deterministic mixture of two markov chains. 

#### Lemma A: 
Let $m_1 = (P_1,V)$ and $m_2 = (P_2,V)$ be not dd but irreducible and finite makrov chains, $P_1 \neq P_2$ and $p_1 \in (0,1)$. 
The random markov chain $M$ is with probability $p_1$, $m_1$ and with probability $1-p_1$, $m_2$. 
Let $x_0,X_1,X_2...$ be a random sequence in $V$, where $x_0 \in V$ is constant and the distribution of $x_0, X_1, ... $ is described by $M$. 

Then for any context length $K \in \mathbb N$:

$ \lim_{T \to \infty} \mathbb P[\exists_{t<T}.\left(\mathbb P[X_{t+1}|H_{t-K:t}] \neq \mathbb P[X_{t+1}|H_t] \right)] = 1 $, 

where $H_{a:b} = X_a, X_{a+1}, ..., X_b$.


#### Proof: 

We first simplify the term $\mathbb P[X_{t+1}|H_{a:t}]$ using the markov assumption:

$\mathbb P[X_{t+1}|H_{a:t}] = \mathbb P[M=m_1|H_{a:t}] \mathbb P[X_{t+1}|X_t, m_1] + \mathbb P[M=m_2|H_{a:t}] \mathbb P[X_{t+1}|X_t, m_2]$

To increase readiblity we denote: $p_{1|H_a} := \mathbb P[M=m_1|H_{a:t}]$ and arrive at:

$\mathbb P[X_{t+1}|H_{a:t}] = p_{1|H_a} P_1(X_t, X_{t+1}) + (1-p_{1|H_a}) P_2(X_t, X_{t+1})$


Know we can simplify this expression:

$ \lim_{T \to \infty} \mathbb P[\exists_{t<T}.\left(\mathbb P[X_{t+1}|H_{t-K:t}] \neq \mathbb P[X_{t+1}|H_t] \right)] = 1 $ 

$\iff  \lim_{T \to \infty} \mathbb P[\exists_{t<T}. (p_{1|H_{(t-K)}} P_1(X_t, X_{t+1}) + (1-p_{1|H_{(t-K)}}) P_2(X_t, X_{t+1}) \neq $

$p_{1|H_{0}} P_1(X_t, X_{t+1}) + (1-p_{1|H_{0}}) P_2(X_t, X_{t+1}))  ]=1$ 



$\iff  \lim_{T \to \infty} \mathbb P[\exists_{t<T}. (p_{1|H_{(t-K)}} (P_1(X_t, X_{t+1})  - P_2(X_t, X_{t+1})) \neq $

$p_{1|H_{0}} (P_1(X_t, X_{t+1}) - P_2(X_t, X_{t+1})))  ]=1$ 



$\overset{a)}{\impliedby}  \lim_{T \to \infty} \mathbb P[\exists_{t<T}.\forall_{t*>t}. (p_{1|H_{(t*-K:t*)}}  \neq p_{1|H_{0:t*}} )  ] = 1$ 

Implication a)  followed from the fact that with probbility 1 there are infinetly many $t$, such that $P_1(X_t, X_{t+1}) \neq P_2(X_t, X_{t+1})$. This is true since $m_1$ and $m_2$ are ireducible.


In order to show the sufficient statement $\lim_{T \to \infty} \mathbb P[\exists_{t<T}.\forall_{t*>t}. (p_{1|H_{(t*-K:t*)}}  \neq p_{1|H_{0:t*}} )  ] = 1$, we will show two things:

- A) $\lim_{t \to \infty} p_{1|H_{0:t}} (= \lim_{t \to \infty} \mathbb P(m_1 | H_t)) \in \lbrace 0,1\rbrace$
- B) For every word $h_K \in V^K$ that can appear in the markov process, e.g. $\exists_t .\mathbb P(H_{t-K:t} = h_K) > 0$ and for every $t$: $p_{1|h_K} := \mathbb P(m_1 | h_k) \in (0,1) $

As there are only finitely many possibilites for $h_k$, from B) follows, that there mus exist $\epsilon > 0$, such that $p_{1|h_K} \in [0+\epsilon, 1-\epsilon]$. From this fact and A) we can conclude the suffitient statement on the righthandside of a).




##### Proof of A)  


Since $P_1 \neq P_2$ there must be $x,y\in V$, such that $P_1(x,y) \neq P_2(x,y)$. Let $T_i$ denote the random variable for the i'th time in which $M$ visits x. As $m_1$ and $m_2$ are irreducible and finite for $i \in \mathbb N$, $T_i$ is finite with probability 1. Note that $\mathbb P[X_{T_i} = x] = 1$. For convenience we denote $p^{(1)} := P_1(x,y)$ and $p^{(2)} := P_2(x,y)$.

Notice that the sequence $X_{T_1 +1}, X_{T_2 +1}, ... | M$ is iid. To make our life easier we define the binomial variable $Y_i = \bold 1[X_{T_i + 1} = y]$. Now $Y_1, Y_2... $ is an infinite sequence of coinflips, that land with probability $p_1$ with probability $p^{(1)}$ on heads and with probability $1-p_1$ with probability $p^{(2)}$ on heads.

Let wlog $p_1 > p_2$. Let $g = (p_1 +p_2)/2$.

$\lim_{t\to\infty}\mathbb P[M = m_1|H_t] \in \lbrace 0,1\rbrace$

$\impliedby \lim_{n\to\infty}\mathbb P[M = m_1|\frac{\sum^n Y_i}{n} > g] \in \lbrace 0,1\rbrace$


$\impliedby \lim_{n\to\infty}\mathbb P[\frac{\sum^n Y_i}{n} > g|M = m_1] \dfrac{p_1}{\mathbb P[\frac{\sum^n Y_i}{n} > g]} \in \lbrace 0,1\rbrace$


$\impliedby \lim_{n\to\infty} 1 \dfrac{p_1}{ p_1 } \in \lbrace 0,1\rbrace$

proof of A) finished

##### Proof of B)

Broof by induction:

Induction base: Let $t_0 \in \mathbb N$:
As $m_1$ and $m_2$ are not dd $\mathbb P[m_1 = M | ()] \in (0,1)$

Induction step: Assume for $0<k< K$, that $\mathbb P[m_1 = M | H_{t_0:t_0+k}] =: p_t \in (0,1)$. Let $t := t_0+k$.

Then $\mathbb P[m = M_1 | H_{t_0:t}, X_{t+1}] = \mathbb P[X_{t+1} | M = m_1,H_{t_0:t}] \dfrac{\mathbb P[M = m_1|H_{t_0:t}]}{\mathbb P[ X_{t+1}| H_{t_0:t}]} $

$= \dfrac{p_t P_1(X_t, X_{t+1})}{p_tP_1(X_t, X_{t+1}) + (1-p_t)P_2(X_t, X_{t+1})} \overset{a)}{\in} (0,1)$


Because $m_1$ and $m_2$ are not dd $P_1(X_t, X_{t+1}) \in (0,1]$ and $P_2(X_t, X_{t+1}) \in (0,1]$, hence a).

Proof of B) finished.


QED

---

Lemma A means the resulting probability distribution of $x_0, X_1, ... $ made in this simple construction can not be expressed by any finite horizon AM model, such as a transformer


----
#### Definition: 
If an autoregressive model with kontext length $K$, $f:\Sigma^K \to \mathcal P_\Sigma $ has the property that $\forall_{h \in \Sigma^{[d]}, x \in \Sigma}f(h)(x)>0$, we call $f$ unrestricted.

----

#### Theorem A.2: 
An upper bounded context $K_t$ length Transformer $T$ can not express the probability distribution a non-deterministic mixture of two unrestricted transformers $(R,J)$ (Romeo and Juliett) with the same token set $\Sigma$ with probability one.

#### Proof: 
We start by showing the possibility of a reduction: Any discrete autoregressive distribution $f$ over alphabet $\Sigma$ with context length $K$, $f:\Sigma^{[K]} \to \mathcal P_{|\Sigma|}$, can be expressed by an finite Markov chain $m=(V_m,P_m)$, where $V_m = \Sigma^{[K]}$. We also show, if $f$ is unrestricted, then $f$ can be expressed by an irreducible makrov chain.

(TODO deal with this K+i shit, somehow allow different kontext lengths)

We construct $P_m$ like this: given $h \in \Sigma^{[K+i]}$, $x \in \Sigma$ and $f(s_K(h))(x) = p$, $P_m((h, s_{K-1}(h) \circ x)) = p$. If a transition $(a,b) \in V_m$ has not defined in the previous sentence (e.g. $s_{K-1}(a) \neq p_{K-1}(b)$), then its porbability is zero $P_m((a,b)) = 0$. Know by construction the $f$ can be fully expressed by $m$:
$f(h)(x) = P_m(h, s(h) \circ x)$. If $f$ is unrestricted it will generate any possible sequence of length $K$ with positive probability.
Therefore $m$ is irreducible.

Let $K$ be the maximum context lenght of R or J and let $m_R = (\Sigma^{[K]}, P_r)$ be the equivalent makrov chain of R and $m_J=(\Sigma^{[K]}, P_j)$ the equivalent makrov chain of $J$.


Let $X_1,X_2...$ be expressed, by a non deterministic mixture of $R$ and $J$.
Lets suppose $T$ could express the probability distribution of this sequence, e.g. $\forall_{t \in \mathbb N} T(H_t)(X_{t+1}) = \mathbb P(X_t|H_t)$. 
a) Then for $t=K_t$, $T$ implements the function $T(H_K)(X_{K_t+1}) = \mathbb P(X_{K_t + 1}|H_K)$.  
b) As $T$ has kontext length $K_t$ it must hold that $T(h)(x) = T(s_{K_t}(h))(x)$. 
From a) and b) follows, that for all $t$, we have $T(H_t)(X_{t+1}) = T(s_K(H_t))(X_{t+1}) = \mathbb P[X_{t+1}| H_{t-K:t}]$.

From lemma A we shall conclude: $ \lim_{T' \to \infty} \mathbb P[\exists_{t<T'}.\left(T(H_t)(X_{t+1}) \neq \mathbb P[X_{t+1}|H_t] \right)] = 1 $

QED 

----
Note that Theorem A.2 doesn't hold, if we have a series of transformers with context window growing to infinity. 

#### Comments: 
We assumed in Theorem A.2 that R,J are unrestricted, e.g. that they take every next token with positive probability. This assumption was necessary for the Theorem to be true to be true in general. Imagine R would give for a word $w$ to be followed by a letter $x$ the probability zero and T not. Then the optimal estimator would disguise T as soon as x would follow after w. Does that mean it stays plausible that for a non deterministic mixture of transfoermes, that are limited to only taking one of the k most likely letters as the next one,  to be implemented as one transformer. I believe only yes, if all parameters of R and J are perfektly well known. As soon as there is some ambiguity in these parameters, suddenlty we can not be exactly sure wich one is among the top k next tokens and suddently again all tokens could appear with positve probability. But thats only a believe and I do not provide any qualitative argument on that.


Eventhough a sparce mixture of Transformers can not be implemented by a finite horizon Transformer, in my opinion this does not reduce the plausibility  of assuming that training data was generated by a sparce mixture of transformers as was done in {{< cite "jeon2024information" >}}. I think this as in my opinion the finite horizon can not capture all details of the text document distribution, as we saw in (todo ref). But it could also be a simplification that doesnt hurt in some scenarios, as is ignoring relativity, when designing an elevator. 


### Does independence suffice for an upper bound of estimation error?
<!---
Lets suppose there was a transformer, that could generate all training documents, that we have. (For this the documents should have an upper bound regarding length. Also the documents should be independent if the transformer is known). 

Now we want to find an upper bound of the estomation error of the obe, but we dont know the bayesian prior, we just
-->

Lets suppose our training ducuments had all an upper bound length $T$, such that there would be a transformer being able to express their distribution, as we assume that the class of transformers can imitate any finite horizon distribution. Lets also assume that for any of these distributions there is a transformer with upper and lower bounded parameters $[-a,a]$.

Now here comes the question: can we use this insight to find an upper bound for any optimal bayesian estimator assitiated to any prior bayesian assumption?

As transformers can express any distribution within our frame and if we start to sample them in the most random way (e.g. highest entropy), should we then get the most random distribution of distributions (e.g. the distribution of distribution, such that we can not find a baesian prior, such that the obe achieves a higher error)? Can we find, what the unifotm distribution is for finite discrete spaces, for AG probability distributions?

If we would find that upper bound, then we would have found an upper bound, which holds independent of the bayesian prior and we would never have to worry again if your assumptions are valid!!! So here is why it fails:

Let $\mathcal P_{\Sigma}$ the set of all probability distributions over tokenset $\Sigma$.
Let $\mathcal P_{T} = \lbrace  \Sigma^{[T-1]} \to P_\Sigma \rbrace$ the set of all autoregressive probability distributions with $T$ tokens. 
An element in $\mathcal P_{T}$ can be seen to take in a sequnce of tokens and to return the probability distribution for the next token.



We call $A: \mathbb R^n \to P_T$ an autoregressive model. If a model $A$ is surjective, we call $A$ a full autoregressive model, since it can parametrize all autoregressive distributions. For parameters $\theta$, we call $A(\theta)$ a parametrized model.

Let $A_t$ represet the transformer architecture, such that $A_t(\theta)$ is a transformer parametrized with $\theta$. $A_t(\theta)(h)(x)$ is then the probability with which this parametrized transformer predicts $x$ after having observed the token sequence $h$. 

As the parameter space is bounded $[-a,a]$, the highest diferential entropy would come from the prior $\Theta \sim \text{unif}([-a,a])^n$ where all parameters are iid.

Intuitively if using $A_t(\Theta)$, should work to create the highest bayesian error, then for any full model $A_x(\Theta)$ this should work as well as there is nothing special with transfomers, as Suvrit Sra put it. For example $A_x$ could be like $A_t(\Theta)$ but first convert $\Theta$ to a gaussian vector. 

As we assumed $A_t$ to be full, the outcome with the highest bayesian error should actually be the entropy of the uniform independent sequence over $\Sigma$ of length $T$, which is $T \cdot \log(d)$, being a higher upper bound as in Theorem Jeon.3.5, which means that a randomly initialized transformer does create a uniform independent token sequence. 


Assuming the training data was generated by an full autoegressive model with random parameters, can be a good assumption to simplify further mathematical analysis and maybe it is not too off, in order to produce results with well meaning and close bounds. Sill choosing random transformers as full autoregressive model, may be a bit arbitrary and I think that further justification would be helpfull to add validity.  

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
Let $\mathcal P_{T} = \lbrace  \Sigma^{[T-1]} \to P_\Sigma \rbrace$ the set of all autoregressive probability distributions with $T$ tokens. 
An element in $\mathcal P_{T}$ can be seen to take in a sequnce of tokens and to return the probability distribution for the next token.

We call $A: \mathbb R^n \to P_T$ an autoregressive model. If a model $A$ is surjective, we call $A$ a full autoregressive model, since it can parametrize all autoregressive distributions. 

At least one full model exists, since $\mathbb R^n$ and $\mathcal P_T$ have the same cardinality.

A transformer is for example an autoregressive model. I dont know whether a transformers with context length T can $\epsilon$-approximate a full autoregressive model.



Let $H_T$ be any random sequence and representented by an autoregressive model $A$, such that its estimation error is bigger than zero. This means there $\mathcal Y \in \mathcal P_{\mathbb R^n}$ and $\theta \sim \mathcal Y$ ($\theta$ be a random vector of some distribution), such that $H_T \sim A(\theta)$ and $\mathbb I(H_T;\theta)>0$.

Notice that $A(\theta) \in \mathcal P_T$. Let $A_f$
be a full autoregressive model . As $A_f$ is surejectiv there is constant $\theta_0 \in \mathbb R^n$, such that $A_f(\theta_0) = A(\theta)$ and thus $H_T \sim A_f(\theta_0)$.

If we model $H_T \sim A_f(\theta_0)$, then $\mathbb H(\theta_0) = 0$ and therefore we obtain an estimation error of $0$: $\mathbb I(H_T;\theta_0) = 0$.

(TODO move definition of full autoregressive model above lemma statement maybe into previous section)

QED

----
I hope reading this  made people, who say that in-context learning is not learning because parameters are not getting updated, think.
(TODO write this down in a good way.)
In Lemma B.1 we found, that we can do such a shitty job when trying to  find a model to express a posterior dstribution, that by looking at the data and updating our model parameters accordingly we will never be able to describe the posterior distribution more precise with the updated parameters, the more tokens we observe, e.g. we will never have to update them. (Still the model would become better at predicting the distribution by observing it, but that information would not be stored in updated model parameters.)

(todo define parametrization)
Can we also do the oposite? Imagine for a given random sequence $H_T$ we can find a parametrization $A, \mathcal Y$, s.t. the estimation error is maximal. Since this parametrization would maximize the estimation error, it would minimize $H(H_T|\Theta)$.
Would we thereby find a model, such that it parameters can describe the sequence the best? Would we therby find the "perfect" model for this sequence? 
Lemma B.2 shows, that for every random sequence there exists a parametrization, such that $H(H_T|\Theta) = 0$, but when you read the proof you will be disappointed.



#### Lemma B.2

Every autoregressive distribution $H_T=X_1, ..., X_T$ can be represented by a model with an estimation error of $\mathbb H(H_T)/T$.


#### Proof

(todo doenst have to be a bijection)

Let $\Sigma$ be the alphabet of $H_T$. Let $f: \Sigma^\infty \to \mathbb R^n$ be a bijection.


We define $A(\theta) := h \to (x \to \mathbb P[f^{-1}$ $(\theta)\_{|h|+1} = x| f^{-1}(\theta)\_{1:|h|} = h]) $

<!---
We define ![Equation](https://latex.codecogs.com/png.latex?A(\theta)%20:=%20h%20\mapsto%20(x%20\mapsto%20\mathbb{P}[f^{-1}(\theta)_{|h|%2B1}%20=%20x%20\mid%20f^{-1}(\theta)_{1:|h|}%20=%20h]))
--->

where $f^{-1}(\theta)_{i:j}$ defines the word that goes from $i$'th cahracter of the word $f^{-1}(\theta)$ until the $j$'th character.
I am sorry that this expression looks confusing but I will try to explain: $f^{-1}(\theta)$ is just an invinite long word. $A(\theta)(h)(x)$ returns 1 iif the word $h \circ x$ is a prefix of $f^{-1}(\theta)$. If $h$ is a prefix of $f(\theta)$, but $h \circ x$, then $A(\theta)(h)(x) = 0$ and otherwise $A(\theta)$ is undefined. $A$ will start to make more sence once $\theta$ is a random variable.

Note that $A$ is a not well-defined autoregressive model, but by far not a full autoregressive model.

Now for a given autoregressive sequence $H_T$, we define the random vector $\theta = f(H_T \circ 0^\infty)$.

Then $A(\theta)(h_t)(x) \overset{a)}{=} \mathbb P(f^{-1} (\theta)\_{t+1} = x | f^{-1}(\theta)\_{1:t} = h\_t) $

$= \mathbb P[X_{t+1}  = x| H_t = h_t]$, e.g. b) $H_T \sim A(\theta)$.


Since $f^{-1}$ is deterministic, $\theta$ contains all information of $H_T$ and we can conclude from a) and b): $\mathbb H(H_T) = \mathbb I(H_T;\theta)$


QED

----


So why are we disapointed after reading the proof of Lemma B.2.? Because eventhough by maximising $\mathbb I(H_T;\theta)$ we found a model that contains all output randomness in its parameters, this model is still shit, as it doesnt permit a way to make more certain probabilistic statements about $\theta$, when observing more elements of the sequence $X_1, ..., X_T$.


So what does the content of this section have to do with its title?
Lemma B.1 and B.2 showed that for any random sequence, we find a way to express this sequence with a model, such that a) no information of the sequence is contained in the model and b) all information of the sequence is contained in the model. Therefore in my opinion the estimation error tells us something about the model, one has choosen to describe a distribution, but less about the distribution itself. I think that from the estimation error we can learn about how much information a given neural architecture can hold in its parameters. At the same time I think that the estimation error tells us less about how well it is possible to learn something from a given distribtion as was done in {{< cite "jeon2024information" >}} (todo add scaling law paper ) (Disclaimer: I didnt look at these papers at such detail to have an established opinion about it, but I will after my exams and update the blogpost accordingly ). I think this as the estimation error can be zero for any distribution as well as the entires distributions entropy, solely depending on how the distribution is modeled. 



## Conclusion

In this blog post we discussed the basics of information theory and presented an example application for data compression with LLMs. Then we discussed the work of {{< cite "jeon2024information" >}} , which aimed at finding an explanation for in-context learning through assumptions on the data generating process. We then discussed the paper and challenged the idea of assuming the data-generation process was made by transformers. We also provided qualitative arguments on that the estimation error says more about a model of data generation, than it says about the distribution of data itself.  Therefore I think that further work has to be done in this analysis to gain insights that are more meaningfull. On the other hand using the information theoretic tool of the estimation error, for analysing neural architectures and scaling laws, as was done in (todo add references), seem promising to me. I didnt look at these papers at such detail to have an established opinion about it, but I will after my exams and update the blogpost accordingly.

# References
{{< bibliography >}} 


https://perchance.org/emoji



The emoji-character sequence from the beginning of the blogpost was generated randomly, after the text of this Blog Post has been written, it is: ⚠️ 📥 😚 🛡 🚦 👹 🌿 

![targets](/figures/gpt_finite_kontext_lenght.png "Assumptio Sparse Mixture") 

