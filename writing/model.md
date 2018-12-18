---
title:  Attention in value-based choice as optimal sequential sampling
author: Fred Callaway
header: Pandoc Example
footer: Callaway
style: Memo
---

\input{commands.tex}

A recent body of work has begun to explore the role of attention (as measured by eye fixations) in decision making. The primary empirical effect is that people are more likely to choose items that they look at more. To explain this effect @Krajbich2010 proposed the attentional drift diffusion model (aDDM), which biases the drift rate in favor of the fixated item. In addition to explaining the basic effect of attention on choice, the aDDM predicts subtler patterns. For example, last fixations are predicted to be shorter than first fixations, because the fixation is cut off when the decision variable crosses a barrier.

[Describe model and describe aditional successful predictions].

Two aspects of the aDDM are 

Here we consider the problem of a decision maker who must choose between one of several goods. The decision maker has limited attentional capabilities and is under some time pressure (perhaps only due to opportunity cost). Given these constraints, how should the agent divide her attention among the competing options? Here we derive a near-optimal solution to this problem and compare to human fixation patterns.

## Problem statement
An agent must choose between $k$ goods, each having some true utility $u_i$. The agent does not have knowledge of these exact values, but she can draw samples from $k$ Normal distributions centered on these true utilities. At some point she chooses one of the items and receives a payout equal to the true utiliy of the chosen item.

We assume sampling has a cost, which may be due to explicit time cost, implicit opportunity cost, internal cognitive cost, or some combination of the three. We additionally assume that _changing_ the focus of attention is costly; that is, it is more costly to sample from a distribution that was not sampled on the last time step. Because we are interested in modeling eye tracking data, we refer to the item that was most recently sampled from as the _fixated_ item; we refer to changing the focus of attention, i.e. sampling from a different distribution as a _saccade_.

## POMDP model
We can model this problem as a partially observable Markov decision process (POMDP) which defines a set of states, actions, and observations as well as functions that determine how action and state determine observations and future states. In our model, state defines the true (unknown) utility and sampling precision of each item as well as the (known) attentional state of the agent i.e. which item is fixated. Actions can be attentional (saccading to a new item) or physical (choosing an item). Observations are noisy estimates of the fixated item's utility. Rewards capture the cost of time/attention, the additional cost of saccades, and the utility of the item that is ultimately chosen. Formally, we define a POMDP $(\S, \A, T, R, \Omega, O)$ where

- $\S = \R^k \times \{1, \dots, k\}$ is the state space. The state is broken up into the true utility of each item $\vec{u}$, and the current fixation $f$.
- hello there
- $\A = \{ \nop, f_1, \dots, f_k , c_1, \dots, c_k \}$ is the action space, where $\nop$ has no effect, $f_i$ saccades to item $i$, and $c_i$ chooses item $i$, ending the episode.
- $T: \S \times \A \rightarrow \S$ is the deterministic transition function that updates only the fixation portion of state based on saccade actions.
- $R: \S \times \A \rightarrow \R$ is the reward function that gives a constant negative reward at each time step, plus an additional negative reward for making saccades. For the choice actions, it gives a reward equal to the utility of the chosen item.
- $\Omega = \R$ is the set of possible observations, which are samples of an item's utility.
- $O: \S \times \A \times \Omega \rightarrow [0, 1]$ is the observation function that gives the probability of drawing a utility sample given the true utility of the fixated item. $O(s, o) = \Normal(o; u_{f}, \sigma)$ where $f$ is the fixated item and $\sigma$ is a free parameter that determines how noisy the samples are.

## Optimal solution

Following Kaelbling et al. (1998), we break down the problem into two parts: a state estimator and a policy. The state estimator maintains a belief (i.e. a distribution over states) based on the sequence of actions and observations. The policy selects the action to take at each time step given the current belief. By combining these two parts, we create an agent that optimally selects fixations and choices given the full history of previous observations.

### State estimator
For ease of exposition, we focus on the belief over item values, noting that the currently fixated item is simply the target of the most recent fixation action. The belief over item values $\vec{u}$ is a multivariate Gaussian with mean vector $\vec{\mu}$ and diagonal precision matrix $\Sigma = \text{diag}(\vec{\lambda})$. Because the observation $o$ is only informative about the currently fixated item (indicated by subscript $f$ below), only the belief about the fixated item changes at each time step. We derive this update by Bayesian inference [cite Murphy?], resulting in


$$
\begin{aligned}
\lambda_f(t+1) &= \lambda_f(t) + \sigma^{-2}  \\
\mu_f(t+1) &= \frac{\sigma^{-2} o + \lambda_f(t) \mu_f(t)}{\lambda_f(t+1)}  \\
\lambda_i(t+1) &= \lambda_i(t) \text{ for } i \neq f  \\
\mu_i(t+1) &= \mu_i(t) \text{ for } i \neq f  \\
\end{aligned}
$$

The belief is initialized to the prior distribution over $\vec{u}$. For now, we assume that the agent knows the true distribution from which utilities are sampled from, which are standard-normal distributed; thus, we have $\vec{\mu}(0) = 0$ and $\vec{\lambda}(0) = 1$.

### Policy
The policy makes two kinds of actions: saccades and choices. We can immediately simplify the problem by reducing the set of choices to a single action, $\bot$, which selects the item that has maximal expected value given the current belief, $\arg\max_i \mu_i(t)$. With this reduction, the problem becomes a _metareasoning_ problem (Hay et al. 2012): at each time step the policy must decide whether or not to gather more information and which (if any) item to gather information about. Such problems are generally impossible to solve exactly due to the infinite (continuous) space of possible beliefs. To address this difficulty, Callaway et al. (2018) proposed a reinforcement learning method for identifying metareasoning policies. They found that their method, Bayesian Metalevel Policy Search (BMPS), found near-optimal policies for a bandit-like metareasoning problem with similar structure to the present problem. Thus, we apply their method and treat the identified policy as "optimal".


<!-- ## Modeling Krajbich & Rangel (2011)
This dataset consists of 3-way choice with eyetracking measures and value ratings. As a first pass, we would like to see if the rational sampling model can capture qualitative trends in the data. To that end, we model the task as a POMDP as described above, with the following parameters (chosen by an informal grid search):

- Utilities $u_i$ are drawn from $\Normal(0, 1)$
- Samples are drawn from $\Normal(u_i, 5^2)$, i.e. $sigma = 5$
- The cost per sample is 0.005
- The (additional) cost of switching is 0.005

Given these parameters, we approximate the optimal policy using BMPS. We then simulate 5000 trials of this policy, drawing utilities from the prior (i.e. we are not yet using the values provided by participants). Based on these simulated trials, we first attempt to roughly recreate the figures from K&R. So far we have reproduced Figures 3A and 3C

![Figure 3A.](figs/3a.pdf){ width=50% }
![Figure 3A.](figs/3a.png){ width=50% }


![Figure 3C.](figs/3c.pdf){ width=50% }
![Figure 3C.](figs/3c.png){ width=50% }


 -->
<!-- Viewing attention as a kind of computation, attention allocation becomes a problem of _metareasoning_, i.e. reasoning about how to allocate limited computational resources. Framing attention in this way allows us to take advantage of formalisms developed for metareasoning, in particular the meta Markov Decision Process (meta-MDP), which treats metareasoning as a sequential decision process. A meta-MDP has the same structure as a traditional MDP ... [more introduction] -->

<!-- Because directing visual attention often involves taking physical actions (i.e. saccades) we modify the standard meta-MDP formalism to account for the interplay between eye movement and belief updates. Formally, we define an MDP $(\S, \A, T, r)$ where a state $s \in \S = \B \times \X$ specifies both the belief $b \in \B$ of the agent and the external state of the agent and the environment -->


<!-- Following previous work, we assume that choices are based on noisy estimates of each item's value, and that these estimates are inferred from noisy samples of the items' true subjective values. However, previous work has generally assumed that the noisy samples have constant and independent gaussian noise (i.e. the gaussian random walk of a drift diffusion model). This  -->


<!-- Typically, the more samples one takes, the more precise the estimates become (although this trend can be violated as discussed below). With this setup, allocating attention corresponds to deciding which item's distribution to sample at each time step. -->



<!-- We formalize the problem of allocating visual attention in preferential choice as a metalevel Markov Decision Process [@huy2012] that describes how attending to different items affects one's beliefs about the values of those items. Concretely, we define a metalevel MDP  $(\B, \C, T, r)$ where

- a belief $b \in \B$ represents independent Normal-Gamma distributions over the mean $\mu$ and precision $\lambdabda$ of each item's value distribuion
- a computatio n $c \in \C$ attends to a single item, samples from its utility distribution, and updates the parameters of the corresponding belief by Bayesian updating
- $T$ is the transition function that specifies the effect of computations  $T(b, c, b') = p(B_{t+1} = b \mid B_t = b, C_t = c)$
- $r$ is the reward function that specifies the cost (negative reward) of attention as well as the utility of the item that is chosen.
 -->





