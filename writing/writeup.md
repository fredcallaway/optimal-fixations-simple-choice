---
title:  Attention in value-based choice as optimal sequential sampling
footer: Callaway
style: Memo
---

\input{commands.tex}

People must often make decisions based on error-prone and incomplete evaluations of the available options. Some errors of judgment may be explained by the lack of access to relevant information; but in many cases, we are not even able to fully take into account the information that _is_ available. This suggests that integrating external information into an internal evaluation requires some cognitive resource that is in limited supply.

A likely candidate for such a cognitive resource is attention. .. TODO ..   

A recent line of work has already begun to explore the role of (visual) attention in value-based decision making. The primary empirical effect is that people are more likely to choose items that they look at more. To explain this effect @Krajbich2010 proposed the attentional drift diffusion model (aDDM), which biases the drift rate in favor of the fixated item. In addition to explaining the basic effect of attention on choice, the aDDM predicts subtler patterns. For example, last fixations are predicted to be shorter than first fixations, because the fixation is cut off when the decision variable crosses a barrier.

Despite these successes, several important questions have not been addressed by previous work. The vanilla drift-diffusion model has been shown to be equivalent to a sequential probability ratio test [citation], providing a rational explanation for why people might use a DDM-like decision mechanism. However, there is no such rational justification for why people would bias their choices with their attention. Indeed, it is hard to imagine how such a bias could possibly improve the quality of choices. Nevertheless, there is some indication that people direct their attention in an adaptive manner: When presented with three or more options, people tend to look more at higher valued items, which in turn makes them more likely to choose those items. Although there is a clear intuition for why people would direct their attention in this manner, there is no formal model of how people allocate their attention in such cases, much less a rational model of how people _should_ allocate their attention.

We attempt to provide answers to both these questions (i.e. why people are more likely to choose items they attend to, and why people allocate their attention towards valuable items) with a single rational model. We begin by defining a Bayesian formulation of the drift-diffusion model, a process of sequential sampling and posterior updating. By endogenizing attention within this model (i.e. allowing an agent to select which item to consider at each time point), we define a sequential decision problem _for how to make a single decision_. 

This problem can be approximately solved by a recently developed reinforcement learning technique [Callaway 2018]. Thus, we can make precise predictions about both how attention should affect choice as well as how attention should be allocated, based on a single rational model.

We apply the model to the dataset of Krajbich & Rangel (2010), in which participants repeatedly selected one out of three possible snacks, each of which was rated in a previous phase of the experiment. At choice time, visual attention was measures by eye tracking. Participants were given as much time as they like to make each choice. This rich dataset allows us to evaluate many different predictions of the model, including how attention affects choice and how attention develops as a function of both item value and previous fixations.

# Computational model

Before formally describing the model, we first provide some intuition. As in the DDM, decisions in our model arise from a process in which evidence for each item is accumulated over time, being tracked by internal "decision variables" that represent the estimated value of each item. In contrast to previous applications of the DDM to non-binary choices, however, we assume that the dynamics of these decision variables are governed by optimal Bayesian inference. Specifically, we assume that the decision maker (or _agent_) tracks Gaussian distributions over the value of each item. When first presented with a choice, each of these distributions is set to a common prior, capturing the distribution of item values across choice problems. As the decision making process progresses, samples are drawn from Gaussian distributions centered on the true (subjective) value of each item. After each sample, the distribution for the sampled item is updated by Bayesian posterior inference, integrating the prior (which may include information from previous samples) with the likelihood provided by each new sample.

The role of attention in this model is to select which item is sampled at each time step. Thus, the problem of deciding where to allocate attention becomes the problem of deciding which distribution to sample from in order to make the most informed and timely decision. Importantly, the agent cannot simply decide to allocate attention to the highest value item because she does not have access to the true values. Rather, she must decide which item to attend to based on her current estimations of each item's value. She may choose, for example, to focus her attention on items that appear to have high value, or perhaps items whose value is still highly uncertain.

Having provided some intuition, we now present a formal model of attention allocation for value-based choice. We model the problem as a partially observable Markov decision process (POMDP), and discuss how we can find a near-optimal solution to the problem using a specially designed reinforcement learning algorithm.

## POMDP model
A partially observable Markov decision precision (POMDP) is defined by a set of states, actions, and observations as well as functions that determine how action and state determine observations and future states. In our POMDP model, the state captures the true (unknown) utility of each item as well as the (known) attentional state of the agent, i.e. which item is currently fixated. The actions can be attentional (fixating on an item) or physical (choosing an item). The observations are noisy estimates of the fixated item's utility. Rewards capture the cost of time/attention, the additional cost of adjusting attention, and the utility of the item that is ultimately chosen. Formally, we define a POMDP $(\S, \A, T, R, \Omega, O)$ where

- $\S = \R^k \times \{1, \dots, k\}$ is the state space where $k$ is the number of items to choose between. The state is broken up into the true utility of each item $\vec{u}$, and the current fixation $f$.
- $\A = \{ f_1, \dots, f_k , c_1, \dots, c_k \}$ is the action space, where $f_i$ corresponds to fixating on item $i$, and $c_i$ corresponds to choosing item $i$, ending the trial.
- $T: \S \times \A \rightarrow \S$ is the deterministic transition function that updates only the fixation portion of state based on fixation actions.
- $R: \S \times \A \rightarrow \R$ is the reward function that gives a constant negative reward at each time step, plus an additional negative reward for attentional shifts (fixating on a different item from the previous time step). For the choice actions, it gives a reward equal to the utility of the chosen item.
- $\Omega = \R$ is the set of possible observations, which are samples of an item's utility.
- $O: \S \times \A \times \Omega \rightarrow [0, 1]$ is the observation function that gives the probability of drawing a utility sample given the true utility of the fixated item. $O(s, f_i, o) = \Normal(o; u_{i}, \sigma)$ where $i$ is the fixated item and $\sigma$ is a free parameter that determines how noisy the samples are. No observation is received for choice actions.


## Metalevel MDP

We now show how we can derive a metalevel MDP from the POMDP model in the previous section. This considerably simplifies the problem of solving the POMDP, allowing us to use previously validated solution methods. A metalevel MDP is formally identical to a standard MDP, i.e. it is defined by a set of states, a set of actions, a transition function, and a reward function; it is distinct from a standard MDP only in its interpretation and the way in which it is derived. In a metalevel MDP, the states correspond to the agent's beliefs, the actions correspond to computations (or cognitive operations), the transition function describes how computations update the agent's beliefs, and the reward function describes the cost of computation, as well as the utility of the item that is ultimately chosen. Readers familiar with the POMDP literature will observe that the metalevel MDP is similar to the belief-MDP representation of a POMDP, with the addition of replacing the choice actions with a single operation that takes the optimal choice given the current belief.

The agent's beliefs are described by a set of Gaussian distributions, one for each item in the choice set. The estimated utility distribution for item $i$ at time point $t$ is $U_i(t) \sim \Normal(\mu_i(t), \lambda_i(t)^{-1})$. The belief can be encoded by two vectors giving the mean and precision of the estimate of each item's value. Thus, the state space of the metalevel MDP is $\R^2k$, where the latter $k$ dimensions are bounded to be strictly positive (because precision is strictly positive).

We now derive the transition function for the metalevel MDP. Because the observation $o$ is only informative about the currently fixated item (indicated by subscript $f$ below), only the belief about the fixated item changes at each time step. We derive this update by Bayesian inference, resulting in

$$
\begin{aligned}
o(t) &\sim \Normal(u_f) \\
\lambda_f(t+1) &= \lambda_f(t) + \sigma^{-2}  \\
\mu_f(t+1) &= \frac{\sigma^{-2} o(t) + \lambda_f(t) \mu_f(t)}{\lambda_f(t+1)}  \\
\lambda_i(t+1) &= \lambda_i(t) \text{ for } i \neq f  \\
\mu_i(t+1) &= \mu_i(t) \text{ for } i \neq f  \\
\end{aligned}
$$.

The belief is initialized to a prior distribution over item values (across all decision problems). We assume that the agent knows the true distribution from which utilities are sampled from. Without loss of generality, we can further assume that they are standard-normal distributed; thus, we have $\mu_i(0) = 0$ and $\lambda_i(0) = 1$ for all $i$.

Having derived the optimal belief updating procedure given fixed observations, we now turn to deriving the optimal decision rule given a belief. Assuming the decision maker must make a choice at time $t$, the decision rule is clear: choose the item that has highest expected value, i.e. $\arg\max_i \mu_i(t)$. This leaves the decision maker with two remaining challenges: When should she stop thinking and make a decision, and which item should she sample from when she decides to think more? Unfortunately, these two challenges do not have an analytical solution when there are more than two items (at least not one that we are aware of). Thus, we turn to approximate methods.

<!-- These are the two primary challenges addressed by  _metareasoning_, a subfield of AI.  -->

The problems of how much to think and what to think about are the primary problems addressed by the field of _metareasoning_ [citations]. Metareasoning is intractable in the general case due to the explosion of possible beliefs that could result from future computations; thus, several approximate metareasoning strategies have been developed. Callaway et al. (2018) proposed a reinforcement learning method for identifying metareasoning policies. They found that their method, Bayesian Metalevel Policy Search (BMPS), found near-optimal policies for a bandit-like metareasoning problem with similar structure to the present problem. Thus, we apply their method and treat the identified policy as "optimal".


## Predictions

The model (comprising both the problem and optimal solution) makes several qualitative predictions about the relationships between attention, value, and choice.


### Value biases attention
In our model, the agent is more likely to attend to items that she believes are valuable. To see why this is rational, consider a case in which there are two items with high and similar value and one item with much lower value. The agent can quickly determine that the low value item is not the best one, and thus does not waste any additional time considering it; thus, the two high-value items receive more attention. In the reverse case, in which there is only one high value item, the agent quickly identifies it, and has no reason to determine which of the similarly low valued items is superior; attention is roughly equally divided. Thus, in net, the positively valued items receive more attention.

Importantly, the true value of items does not directly influence attention; this effect is mediated by the internal value estimates that the agent builds up over the course of a decision. In the initial stages of a decision, the agent's value estimates will be highly uncertain and noisy, with a weak dependence on the true values. Thus, although she will tend to attend to the item she _believes_ is most valuable, it is likely that this item is not truly the most valuable. As the decision progresses, these estimates will become progressively less noisy. Thus, the estimated values that are biasing attention will more closely align with the true values. As a result, we expect to see that the tendency to attend to high value items should increase over the course of the decision.

Note that the attentional bias only holds for decisions between three or more items. Indeed, it has been shown that attention should be exactly evenly divided in the two alternative case [@Fudenberg2018] and our model shows this behavior. Accordingly, no effect of value on attention has been found in two-alternative choice experiments [citations].


### Attention biases choice
In our model, the more an agent attends to an item, the more likely she is to ultimately choose it. This relationship holds even if we hold the true values constant. Thus, we predict that people will be more likely to choose an item that they looked at more, accounting for any difference in the ratings previously assigned to each item. This prediction is also made by the aDDM, a direct result of the assumption that the drift rate is positively biased towards the attended item. However, our model makes the same prediction without any assumption of biased sampling.

There are two mechanisms through which the effect can emerge in our model. The first mechanism is through a mismatch between the prior distribution and the true distribution from which items are drawn. If the mean of the true distribution is higher than the mean of the prior, then sampling from an item will on average increase its estimated value. Thus, as in the aDDM, the prediction arises from a causal (positive) effect of attention on choice, mediated by estimated value.

The second mechanism by which the apparent attentional bias may emerge depends on the rational allocation of attention. Under this explanation, there is no positive effect of attention on value. Instead, there is a positive effect of value on attention. That is, the rational model predicts that (in the case of three or more items), the agent will be more likely to attend to an item that she believes is valuable. At the same time, she is also more likely to choose items that she believes are valuable. Importantly, both choice and attention allocation depend on _estimated_ value, not the true item values; thus, the causal dependence is not broken by fixing (conditioning on) true value.

```
                  Possible figure

Mechanism #1: attention -> estimated value -> choice

Mechanism #2: attention <- estimated value -> choice
```