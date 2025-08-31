# some formulas used somewhere in the code

I found most things I needed [here](https://en.wikipidia.org/wiki/Backpropagation#Derivation)

## gradient descent

$$\large w_{neu} = w_{alt} + \Delta w$$

$$\large \Delta w =-\mu \delta o$$

## loss functions

mean squared error:

$$\large \frac{1}{2} \sum_{i=1}^{n} (t_{i} - o_{i})^2$$, 
where n is the number of output neurons

### Cross-entropy
> The cross-entropy between two probability distributions $p$ and $q$, over the same underlying set of events, measures 
> the average number of bits needed to identify an event drawn from the set when the coding scheme used for the set is 
> optimized for an estimated probability distribution $q$, rather than the true distribution $p$. \
> *wikipedia.org*

Let's dissect this definition!
Cross-entropy comes from information theory, which studies the quantification, storage and communication of information.
In the scope of information theory entropy is a key measure used to quantify the amount of uncertainty in the value of a
random variable.
A coding scheme is a way to encode events of a distribution $p$ in bits. Ideally, the encoding of events uses as little
bits as possible. Hence, the events that are more likely should be encoded with less bits and vice versa. For example,
consider following coding scheme:

the set of underlying events: $\chi = {A, B, C}$ 

$$\large \sum_{x \ \in \ \chi} \ p(x) \ log \ q(x) $$

Since $p(x)$ is either 1 or 0 for binary classification this equation can be reduced to:

$$$$