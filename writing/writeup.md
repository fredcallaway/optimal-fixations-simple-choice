---
title:  Attention in value-based choice as optimal sequential sampling
author: Fred Callaway
footer: Callaway
style: Memo
---


$f(x)$

\input{commands.tex}

People must often make decisions based on error-prone and incomplete evaluations of the available options. Some errors of judgement may be explained by the lack of access to relevant information; but in many cases, we are not even able to fully take into account the information that _is_ available. This suggests that integrating external information into an internal evaluation requires some cognitive resource that is in limited supply.

A likely candidate for such a cognitive resource is attention. ... TODO ...

A recent line of work has already begun to explore the role of (visual) attention in value-based decision making. The primary empirical effect is that people are more likely to choose items that they look at more. To explain this effect @Krajbich2010 proposed the attentional drift diffusion model (aDDM), which biases the drift rate in favor of the fixated item. In addition to explaining the basic effect of attention on choice, the aDDM predicts subtler patterns. For example, last fixations are predicted to be shorter than first fixations, because the fixation is cut off when the decision variable crosses a barrier.

Despite these successes, several important questions have not been adressed by previous work. The vanilla drift-diffusion model has been shown to be equivalent to a sequential probability ratio test [citation], providing a rational explanation for why people might use a DDM-like decision mechanism. There is no such rational justification for why people would bias their choices with their attention; indeed, it is hard to imagine how such a bias could possibly improve the quality of choices. Nevertheless, there is some indication that people direct their attention in an adaptive manner: when presented with three or more options, people tend to look more at higher valued items, which in turn makes them more likely to choose those items. Although the intuition for why people might direct their attention in this manner is clear, there is no formal model of how people allocate their attention in such cases, much less a rational model of how people _should_ allocate their attention.

We attempt to provide answers to both these questsions (i.e. why people are more likely to choose items they attend to, and why people allocate their attention towards valuable items) with a single rational model. We begin by defining a Bayesian formulation of the drift-diffusion model as a process of sequential sampling and posterior updating. By endogenizing attention within this model (i.e. allowing an agent to select which item to consider at each time point), we define a sequential decision problem _for how to make a single decision_. The resulting _metalevel Markov decision process_ can be approximately solved by a recently developed reinforcement learning technique [Callaway 2018]. Thus, we can make precise predictions about both how attention should affect choice as well as how attention should be allocated, based on a single rational model.

We apply the model to the dataset of Krajbich & Rangel (2010), in which participants repeatedly selected one out of three possible snacks, each of which was rated in a previous phase of the experiment. At choice time, visual attention was measures by eye tracking. Participants were given as much time as they like to make each choice. This rich dataset allows us to evaluate many different predictions of the model, including how attention affects choice and how attention develops as a function of both item value and previous fixations.

# Computational model

Before formally describing the model, we first provide some intuition. Like the DDM, our model treats decisions as arrising from a process in which evidence for each item is accumulated over time, being tracked by internal "decision variables" that represent the estimated value of each item. In contrast to previous applications of the DDM to non-binary choices, we asume that the dynamics of these decision variables are governed by optimal Bayesian inference. Specifically, we assume that the decision maker tracks Gaussian distributions over the value of each item. When first presented with a choice, each of these distributions is set to a common prior, capturing the distribution of value of items across choice problems. As the decision progresses, samples are drawn from a Gaussian distribution centered on the true (subjective) value of each item; the decision-specific distributions are updated by Bayesian posterior inference, integrating the prior (which may include information from previous samples) with the likelihood provided by each new sample.

We assume that only one sample is drawn at each time step, and that this sample is drawn from the currently attended item. We assume that the decision maker has control over her attention, that is she can decide which item to sample from at each moment. The decision about what item to sample at each time point is made based on the distributions over item values at that time point. Initially, these distributions are identical, and thus attention will be allocated randomly. However, as the decision progresses, the agent may choose to focus her attention on items that appear to have high value, or items whose value is still highly uncertain. Importantly, the agent cannot simply decide to allocate attention to the highest value item because she does not have access to the true values.

Having provided some intuition, we now present the formal model of value-based choice with directed attention. We first model the problem as partially observable Markov decision process (POMDP) which defines a set of states, actions, and observations as well as functions that determine how action and state determine observations and future states. We then derive a metalevel Markov decision process (metalevel MDP) from this POMDP, and discuss our approximate solution method.

## POMDP model
In the POMDP model, the state captures the true (unknown) utility of each item as well as the (known) attentional state of the agent, i.e. which item is currently fixated. The actions can be attentional (saccading to a new item) or physical (choosing an item). The observations are noisy estimates of the fixated item's utility. Rewards capture the cost of time/attention, the additional cost of saccades, and the utility of the item that is ultimately chosen. Formally, we define a POMDP $(\S, \A, T, R, \Omega, O)$ where

- $\S = \R^k \times \{1, \dots, k\}$ is the state space. The state is broken up into the true utility of each item $\vec{u}$, and the current fixation $f$.
- $\A = \{ f_1, \dots, f_k , c_1, \dots, c_k \}$ is the action space, where $f_i$ corresponds to fixating on item $i$ , and $c_i$ corresponds to choosing item $i$, ending the trial.
- $T: \S \times \A \rightarrow \S$ is the deterministic transition function that updates only the fixation portion of state based on fixation actions.
- $R: \S \times \A \rightarrow \R$ is the reward function that gives a constant negative reward at each time step, plus an additional negative reward for making saccades (fixating on a different item from the previous time step). For the choice actions, it gives a reward equal to the utility of the chosen item.
- $\Omega = \R$ is the set of possible observations, which are samples of an item's utility.
- $O: \S \times \A \times \Omega \rightarrow [0, 1]$ is the observation function that gives the probability of drawing a utility sample given the true utility of the fixated item. $O(s, f_i, o) = \Normal(o; u_{i}, \sigma)$ where $i$ is the fixated item and $\sigma$ is a free parameter that determines how noisy the samples are. No observation is received for choice actions.

## Metalevel MDP model

We now show how we can derive a metalevel MDP from the POMDP model in previous section. This considerably simplifies the problem of solving the POMDP, allowing us to use previously validated methods. A metalevel MDP is formally identical to a standard MDP, i.e. it is defined by a set of states, a set of actions, a transition function, and a reward function. A metalevel MDP is distinct from a standard MDP only in its interpretation and the way in which it is derived. In a metalevel MDP, the states correspond to the decision maker's beliefs, the actions correspond to computations (or cognitive operations), the transition function describes how computations update the agent's beliefs, and the reward function describes the cost of computation, as well as the utility of the item that is ultimately chosen. Readers familiar with the POMDP literature will observe that the metalevel MDP is similar to the belief-MDP representation of a POMDP, with the addition of replacing the choice actions with a single operation that takes the optimal choice given the current belief.

The agent's beliefs are described by a set of Gaussian distributions, one for each item in the choice set. The estimated utility distribution for item $i$ at time point $t$ is $U_i(t) \sim \Normal(\mu_i(t), \lam_i(t)^{-1})$. Thus the belief can be encoded by two vectors giving the mean and precision of the estimate of each item's value. Because the observation $o$ is only informative about the currently fixated item (indicated by subscript $f$ below), only the belief about the fixated item changes at each time step. We derive this update by Bayesian inference [cite Murphy?], resulting in


$$
\begin{aligned}
\lambda_f(t+1) &= \lambda_f(t) + \sigma^{-2}  \\
\mu_f(t+1) &= \frac{\sigma^{-2} o + \lambda_f(t) \mu_f(t)}{\lambda_f(t+1)}  \\
\lambda_i(t+1) &= \lambda_i(t) \text{ for } i \neq f  \\
\mu_i(t+1) &= \mu_i(t) \text{ for } i \neq f  \\
\end{aligned}
$$

<!-- TODO, update all below -->

The belief is initialized to the prior distribution over $\vec{u}$. For now, we assume that the agent knows the true distribution from which utilities are sampled from, which are standard-normal distributed; thus, we have $\vec{\mu}(0) = 0$ and $\vec{\lambda}(0) = 1$.

### Policy
The policy makes two kinds of actions: saccades and choices. We can immediately simplify the problem by reducing the set of choices to a single action, $\bot$, which selects the item that has maximal expected value given the current belief, $\arg\max_i \mu_i(t)$. With this reduction, the problem becomes a _metareasoning_ problem (Hay et al. 2012): at each time step the policy must decide whether or not to gather more information and which (if any) item to gather information about. Such problems are generally impossible to solve exactly due to the infinite (continuous) space of possible beliefs. To address this difficulty, Callaway et al. (2018) proposed a reinforcement learning method for identifying metareasoning policies. They found that their method, Bayesian Metalevel Policy Search (BMPS), found near-optimal policies for a bandit-like metareasoning problem with similar structure to the present problem. Thus, we apply their method and treat the identified policy as "optimal".

## Predictions
Unlike in the aDDM, this sample is not positively biased---it is drawn from a distribution centered on the attended item's true value. However, if that item's value is greater than the initial prior mean, the update will be positive on average, i.e. the posterior mean will be higher than the prior mean in expectation. 
<!--  -->