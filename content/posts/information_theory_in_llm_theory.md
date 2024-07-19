---
title: Information Theory in LLM Theory
bibFile: data/bibliography.json # path relative to project root
---

⚠️ 📥 😚 🛡 🚦 👹 🌿 *By reading this blogpost, you will find out, why this emoji sequence is here.*


# Introduction

# Background

 

### A Brief Introduction on Information Theory

![targets](/figures/venn_information.png "Assumptio Sparse Mixture") 


In this section we want to provide a brief overview on Information Theory. 

Imagine you lived $40\cdot n$ years in the future and that the population has doubled every $40$ years. For simplicity assume that in the current present just one person is alive. Conclusively $2^n$  people live at your point in the future. As you live very far in the future $n$ is very large, about $2^{10^{15}}$. "Today" is your friends birthday party and you dont have a present. In order to avoid an embarassing moment you decide to hire a hitman to kill his neighbour, such that the party will be postponed. In order to do this you have type in the id of that person (which is one Pentabit long) into the hitmans website and he completes the task within the same day. After you received a message that the birthdayparty is postponed you are very relieved, but then suddently you start wondering. You only have a 1Gbit/s upload speed (hardware did not improve so much in the last $40\cdot2^{10^{15}}$ years), how have you been able to upload $10^{15}$ bits within just one day? This seems impossible as there are $2^{10^{15}}$ people and thus the id is $10^{15}$ bits long. Then you find out why: the website did not have to send $10^{15}$ bits of information to its server. Small parts of the information have already been known by the server. For example the information that people in your region are more likely to order a kill for someone in that same region. 


Information theory is the study on expressing the quantity of information called entropy. Entropy of an information is the minimal espected amount of bits (todo explain that it doesnt have to be bits) required to encode this information.

One can show, that the entropy $\mathbb H$ in bits of a discrete random variable $X$ with a known probability density function $p_X$ is

$\mathbb H(X) = \sum_{x \in \mathcal X} - p_X(x) \log_2(p_X(x)) = \mathbb E[-\log_2 X]$

Often instead of the binary logarithm the natural logarithm is used to express entropy. The resulting unit is called nats instead of bits. 

$\mathbb H(X) = \sum_{x \in \mathcal X} - p_X(x) \log(p_X(x))$

The entropy of continoous random variables is infinity, as they can have infinetly many outcomes (TODO provide source). But the differential entropy $\bold h$ of a continous random variable. (TODO explain use of diferential entropy)


$\bold h(X) = \mathbb E[-\log(X)]$


Information theory provides several expressions for how the information of different random variables relate.  This is useful in many different scenarios. (<- TODO example)
Definition:

- Joint Entropy: $\mathbb H(X,Y) := \mathbb H((X,Y))$
- Conditional Entropy: $\mathbb H(X|Y) := \mathbb E[(y \to \mathbb H(X|Y=y))(Y)] = \sum_{y \in \mathcal Y} p_y(y) \sum_{x \in \mathcal X} p_{X|Y=y}(x) - \log(p_{X|Y=y}(x))$ (<- todo chec correct)
- Mutual Information: $\mathbb I(X;Y) := \mathbb H(X) - \mathbb H(X|Y)$


$\mathbb H(X,Y)$ is just the entropy of the joint variable $(X,Y)$. $\mathbb H(X|Y)$ is the expected entropy of $X$ if $Y$ is known. And $\mathbb I(X;Y)$ completes the elements of the ven diagram. It is the expected gain of information on $X$ (with gain of information I mean reduction of entropy) if $Y$ is known and vice versa, as mutual information is kommutative (see TODO insert venn diagramm). 

Please note that on Wikipedia equivalent but different definitions of these terms are stated. I choose these expressions as the definitions, as they are a) better to get an intuitive understaning and b) seem resemble more what the inventor had in mind, when coming up with these definitions. 


<!---
- briefly name an example applications from coding theory
- briefly name an example of neurology where neurons may maximize their entropy
-->



### Philosofical Interlude: Does Information Exist?

Chapter Draft (everything in this draft is very vague, dont read)
- make reader know that you are critic of your own ideas
 - Previous Section information relied on an assumed probability space
 - what is probability? It is a way to formally express of what can be known and what can not be known? (<- express waeknasses of that thought and derive it and be more precise and present alternatives of expressong of what can be known and what not)
- Problems of probability:
    - How well can be known what can not be know? Well enough to express probabilitys? (name logic example to define what can be known and what can be not and probabilistic example, and argue whether there is something inbetween)

    - maybe also everything can be known and everything happens with probability one (<- drop argument, why one can not be sure that universe is not deterministic universe, by using argument based on an explanation why even anything exists)
        - explain that under that circumstances only physically isolated can be known, using similar "proof" as holding theorems proof
        - include limitations of calculation power into probability?
- use previous arguments to express where probabilistic assumptions might  be off and name examples on how it could affect us practically
- conclude if we are carefull enough with probabilistic assumptions under scenario a everything has information 0, under scenario b everything has information $\infty$ and under scenario a using limitation of physical possible calculation power information content is impossibly hard to compute
- mention that information theory has been usefull anyway



## The Link of Information Theory and LLMs


Information theory can be found everywhere, wherethere is uncertainty


## Analyzing Neural Network Architectures {#anal_nn}


![targets](/figures/assumption_for_ideal_bayesian_estimator.png "Assumption AM") 


In this chapter results from {{< cite "jeon2022information" >}} and {{< cite "jeon2024information" >}} will be discussed. For several generative models models upper bounds have been established of how much information they can generate. These results can be used make statements about the expressiveness of different neuronal architecture. (<- be more precise about expressivemness)



# An Information Theoretic Perspective Analysis on In-Context Learning

In {{< cite "jeon2024information" >}} assumptions about the origin of the trianing data of LLMs have been made from which an explanation of in context learning was stated.


## What is In-Context Learning


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


You might remember that earlier in this chapter I said that the authors assumed, that all training data has been generated by a transformer. And now I suddently presented a sparse mixture of transformers instead of a transformer. This is because in the conclusion the authors said, that they hope, how further mathematical analysis will be able di describe how a transformer can implement a sparce mixtrue of transformers. So actually they did not make this assumption, but this assumption might be made in the future once it has been prooven that a transformer can implement a sparce mixtre of transformers.

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

$\mathcal L_{M,T} = \dfrac{\mathbb I(H_T^{(M)};\theta)}{MT}$


Proof:

We know $\mathbb H(X) = \mathbb H(X|Y) + \mathbb I(X;Y)$, therfore it suffices to show that $\mathbb L_{M,T} \cdot MT = \mathbb H(H_T^{(M)})$. This is true since:

$\sum_{m=1}^{M} \sum_{t=0}^{T-1} \mathbb E[ - \ln \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]] = \mathbb E[-\ln \prod_{m=1}^{M} \prod_{t=0}^{T-1} \mathbb P[X_{t+1}^{(m)}|H_t^{(m)}]] \overset{a)}{=} \mathbb E[-\ln  \mathbb P[H_T^{(M)}]]= \mathbb H(H_T^{(M)})$


a) follows from the chain rule of probability.

QED

Be  aware I often do mistakes, when proofing something.
I presented my own version of the proof of this equation as it contains the intermediate result $\mathbb L_{M,T}  = \dfrac{\mathbb H(H_T^{(M)})}{MT}$. Thus we can see $\mathbb L_{M,T}$ as the average entropy per token.


This equation means roughly speaking, that the estimation error consits of these parts in $\theta$, that are conveyed to $H_T^{(M)}$. For example as we often model $\theta$ as a continous random variable and $H_T^{(M)}$ as a discrete random variable, $H_T^{(M)}$ can not contain all information in $\theta$. (<- be more clear)

As in previous section, we have worked out a way to separate $\theta = \theta_1, ...\theta_m$ into two independent random variables, namely $\theta|\psi$ and $\psi$, we continue by expressing $\mathcal L_{M,T}$ with these two random variables. 


$\mathcal L_{M,T} = 
\underbrace{\dfrac{\mathbb I(H_T^{(M)};\psi)}{MT}}_\text{meta
estimation error} + 
\underbrace{\dfrac{\mathbb I(D_m;\theta_m|\psi)}{T}}_\text{intra document estimation error}$


To proof this equation it suffices to show 

- A) $\mathbb I(H_T^{(M)};\theta) = \mathbb I(H_T^{(M)};\psi) + \mathbb I(H_T^{(M)};\theta|\psi) $
- B) $\mathbb I(H_T^{(M)};\theta|\psi) = M \cdot \mathbb I(D_m;\theta_m|\psi)$


We defined earlier $D_m \bot \psi | \theta_m$, which means $ H_T^{(M)} \bot \psi | \theta$ (<- be more formal?). Therefore $\mathbb I(H_T^{(M)};\theta) = \mathbb I(H_T^{(M)};(\theta, \psi))$. Then A) follows from the chain rule of mutual information.

B) follows from $\mathbb I(H_T^{(M)};\theta|\psi) \overset{a)}{=} \sum_{m=1}^M  \mathbb I(D_m;\theta_m|\psi) \overset{b)}{=} M\cdot \mathbb I(D_m;\theta_m|\psi) $. 

Equation a) holds because the pairs $(D_1, \theta_1)|\psi, ..., (D_M, \theta_M)|\psi$ are independent and mutual information is additive for independent variables.(http://www.scholarpedia.org/article/Mutual_information)  
As $ (D_1, \theta_1),..., (D_M, \theta_m) | \psi$ are identically distributed, for any $a,b \in \{1,...,M\}$, $\mathbb I(D_a;\theta_a|\psi) = \mathbb I(D_b;\theta_b|\psi)$. Therefore b) is true.



(<- proof not peer reviewed)
QED  


The authors seperate the term into two parts, the meta estimation error and the intra document estimation error. The meta estimation error describes the error, of learning information shared shared by all documents. The intra document estimation error is the error of learning the parameters of learning individual documents after having learned the shared information $\psi$. Lets say we have arbitrarily many training samples $A$ for our optimal bayesian estimator in which $D_m$ does not occur. Due to them being arbitrarily many, let's suppose $\psi$ has been well enough discovered, such that $\mathbb I(D_m;\psi | A)$ is about 0. Then the error of estimating $D_m$ is about the intra-document error. This means, that even of we perfectly train our model, there will be some error bigger than zero related to the estimation of the probability that generates the data? Or will it realy be bigger than zero? More on this cliffhanger later (<- TODO!) 


### Results from assumptions of bayesian prior

In order to make the results about the error of the optimal bayesian estimator more concrete more assumptions about the bayesian prior are required. Unfortinately as discussed in section (TODO) our view on amount of information requires the distribution of the bayesian prior to be well enough known (in this case "well enough" usually means ""). In the previous chapter 




## Discusion of their Assumptions {#discussion-assumptions}

### Can we assume the existance of a transformer, generating all training documents

- transformers can imitate every upper bounded finite horizon am model
- but they cant imitate infinite horizon documents
- claim A: there exists at least one document, whichs very end is dependent on the very beginning of the document, even if the middle of the document is known
- proof of claim A is written at the very end of this document

### Can a transformer implement a sparse mixture of transformers?

No a sparce mixture of transformers has an infinite horizon

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


### Does the estimation error have a meaning

- write down how every distribution can be represented with an estimation error of 0 (use functions A and K, use uniform instead of gausian bayesian prior)

- write how cool it would be if we where able to find for a given X distribution, the bayesian prior that maximises the estimation error
  - would the resulting model then be the perfect model for this distribution?
    - this is the model that would minizie H(X|Theta)   
    - this means that for many samples better estimation performance? (no bayesian prior always perfekt, but we are not, maximal possible estimation error tells us how much randomness will be removed for many samples) 
  - for example uniform distribution forces estimation error to be zero, e.g. every modelling of uniform distribution is equal as shit




|||||||| Below Unfinished Ideas |||||||

#### When can we make assumptions of the the familie of AMs

Lets suppose following scenario: we have a sequence of random tokens $X_1, ..., X_T$ and know that this sequence is autoregressive with finite horizon T-d, which means $\mathbb P(X_{t+1} | X_{t-T+d}, ..., X_{t-1}) = \mathbb P(X_{t+1}|H_t, X_{t+2}, ...X_{T})$.
Then, the distribution of $H_T$ can be charachterized by the function $f(H_t) = \mathbb P(X_{t+1} | H_t)$. Thus there is a trivial bijection from the set $\{ \Sigma^{T-d} \to \mathcal P_\Sigma \}$ to the Autoregressive distributions over the alphabet $\Sigma$ and length $T$ and horizon $T-d$. (<- warning you have to integrate the non character sign to sigma and that can only be place to the right position (change $\Sigma^T$)).

Lets suppose we are reely lucky and  find a parmatrization of all autoregressive distributions of horizon T-d by finding an injektive function $A :\mathbb J_\sigma^n \to \{\Sigma^{T-d} \to \mathcal P_\Sigma \}$, where $\mathbb J_\sigma ^n := \{||x||_2 = \sigma, \mathbb{1} ^T x=0 | x \in \mathbb R^n\}$. (injecion exist since in and output sets are the same size)

(

Let now $\Theta \sim \mathcal N(0,I_n\sigma)$. Then $A(\Theta)$ is a random random distribution which itself is a random distribution. This means $A(\Theta) \in \{\Sigma^T \to \mathcal P_\Sigma \}$ (Note $X \sim \mathcal N(0,1) \to x \notin \mathbb R$) 



<!--This means there must exits $\theta \in \mathbb J^n$, such that  $A(\theta) = A^{-1}(A(\Theta))$-->

<!--Try to find argument that autogenerative model can not create itsel, can not be a finite horizon thing, but I think that is going nowhere-->


If A exists, and we  insert random vector into A, then that random variable is element of As image

for every random bayesian prior there exist a constant bayesian prior!!

(!!for every constant bayesian prior  model there exists another model with non constant bayesian prior (just add another theta i and make x independent of it, trivial bayesian + model modification)!!!)

but not for every distribution there is a bayesian prior, such that estimation error bigger than zero (<- thats the actual inverse)hehehe



lets find K that maps every distribution of theta to its kernel theta

e.g. K: random distribution -> constant

$A(K(l(\Theta)) = A(\Theta)$

<!--Say that iid if theta is not the worst case assumption for our model.-->

To recapulate, we assumed/showed evidence, that there can exist a pair $A_t,\theta$ that can generate any upper bounded horizon distribution, when $\theta$ is empirical gaussian and $A_t$ is a parametrization function for transformers. Lets suppose we wanted to use that knowledge to find an upper bound for the error of the optimal bayeisan optimizer e.g.  an upper bound of the mutual information H_t Theta. Can we do that by using the random random distribution $\Theta \sim \mathcal N(0,\sigma)$, as that way $\Theta$ is the probability distribution with maximal differential entropy? 
<!-- Not sure yet whether this is the actual problem -->
The problem is, that there exists $\theta \in \mathbb J^n$, s.t. $A(\theta) = A(\Theta)$.

The problem is, that this assumes, that $A$ is the only function (it is actually class of function) that can do that (explain what). Why
 suppose B (b entagles and makes dependencies) 



)
!!!!!!
But $A(\theta)$ only has horizon of $K<T$ and $A(\Theta)$ has infinite horizon (if A is paramatrization of transformers) and thus models and thus A can only exist for AMs with horizin at least T or with AMs that do not increase the horizon.  


This means only a bayesian estimator with horizon T can detect $\Theta$??


conclusion from A: for each prior distribution of Theta there is a constant theta that creates a prediction error of 0 but the same distribution of X_t



can A be inverted for arbitrarili long horizon case? by definition yes. so also for unbounded case we find good and shit model?

## Conclusion

In this blog post several applications of Information Theory in Deep Learning and LLMs where presented. 

## Take home messages


# References
{{< bibliography >}} 


https://perchance.org/emoji



Proof of claim A: (<- TODO link claim A>)
The emoji-character sequence from the beginning of the blogpost was generated randomly, after the text of this Blog Post has been written, it is: ⚠️ 📥 😚 🛡 🚦 👹 🌿 