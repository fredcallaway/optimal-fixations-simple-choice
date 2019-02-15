---
title:  Attention in value-based choice as optimal sequential sampling
footer: Callaway
style: Memo
---

\input{commands.tex}

Consider a diner, seated in a restaurant she has never eaten at, perusing the menu to decide what she wants to have for dinner. Under the standard model of economic decision making [@Rangel2008; @Kahneman1979], she would assign each item a value and choose the one with maximal value. While her preferences might change day to day (leading to inconsistent choices), she always chooses the item that she believes is best. Unfortunately, this idealization does not well capture the experience many of us have. Instead, we undergo a difficult and sometimes lengthy process of weighing the options, initially being drawn to one entree before identifying a competitor, osscilating between them, failing to notice a desirable choice, and feeling pangs of regret at the sight of our companion's meal.

A great deal of work in psychology, neuroscience, and behavioral economics has attempted to better capture the process through which people make decisions. A key insight from psychology and neuroscience is that decision-making is a sequential process. This insight is formalized in the drift-diffusion model [@Milosavljevic2010], where the osscilation between mushroom risotto and pesto gnocchi is captured by the perturbations of a random walk.

<!-- Another key insight, coming from enonomics is that ... rational inattenion[@Matejka2015] -->


<!-- People often make decisions based on error-prone and incomplete evaluations of the available options. Some errors of judgment may be explained by the lack of access to relevant information; but in many cases, we are not even able to fully take into account the information that _is_ available. This suggests that there are limitations on our ability to pr\usepackage{lastpage}
\newpagestyle{fancy}{
  \setfoot{}{\thepage \ of \pageref*{LastPage}}{}
  \sethead
    { Attention in value-based choice as optimal sequential sampling }
    {}
    { Callaway }
  \headrule
  \setheadrule{0.3pt}
}
\pagestyle{fancy}

\title{\vspace{-2em}Attention in value-based choice as optimal sequential sampling}

\date{\vspace{-1em}}

\begin{document}
\maketitle


\input{commands.tex}

A recent line of work has already begun to explore the role of (visual)
attention in value-based decision making. The primary empirical effect
is that people are more likely to choose items that they look at more.
To explain this effect, \citet{Krajbich2010} proposed the attentional
drift diffusion model (aDDM), which biases the drift rate in favor of
the fixated item. In addition to explaining the basic effect of
attention on choice, the aDDM predicts subtler patterns. For example,
last fixations are predicted to be shorter than first fixations, because
the fixation is cut off when the decision variable crosses a barrier.

Despite these successes, several important questions have not been
addressed by previous work. The vanilla drift-diffusion model has been
shown to be equivalent to a sequential probability ratio test
\citep{Bogacz2006, Bitzer2014}, providing a rational explanation for why
people might use a DDM-like decision mechanism. However, there is no
such rational justification for why people would bias their choices with
their attention. Indeed, it is hard to imagine how such a bias could
possibly improve the quality of choices. Nevertheless, there is some
indication that people direct their attention in an adaptive manner:
When presented with three or more options, people tend to look more at
higher valued items, which in turn makes them more likely to choose
those items. Although there is a clear intuition for why people would
direct their attention in this manner, there is no formal model of how
people allocate their attention in such cases, much less a rational
model of how people \emph{should} allocate their attention.

We attempt to provide answers to both these questions (i.e.~why people
are more likely to choose items they attend to, and why people allocate
their attention towards valuable items) with a single rational model. We
begin by defining a Bayesian formulation of the drift-diffusion model, a
process of sequential sampling and posterior updating. By endogenizing
attention within this model (i.e.~allowing an agent to select which item
to consider at each time point), we define a sequential decision problem
\emph{for how to make a single decision}. We formalize this problem as
\emph{metalevel} Markov decision process, and identify an approximately
optimal solution using a recently developed reinforcement learning
technique \citep{callaway2018learning}. Thus, we can make precise
predictions about both how attention should affect choice as well as how
attention should be allocated, all within a single rational model.

We apply the model to the dataset of Krajbich \& Rangel (2010), in which
participants repeatedly selected one out of three possible snacks, each
of which was rated in a previous phase of the experiment. At choice
time, visual attention was measures by eye tracking. Participants were
given as much time as they like to make each choice. This rich dataset
allows us to evaluate many different predictions of the model, including
how attention affects choice and how attention develops as a function of
both item value and previous fixations.

\section{Computational model}\label{computational-model}

Before formally describing the model, we first provide some intuition.
We treat decision making as an iterative process of sampling and
inference. The decision maker (or \emph{agent}) is presented with a set
of items, each of which has some true unknown value. In order to
determine which item to choose, the agent can generate noisy samples of
each item's utility, each sample providing a small amount of information
about the utility of a single item, but also having a small cost. The
agent continously integrates information from these samples (along with
a general prior over item value), thus developing an increasingly
precise and accurate belief about each item's value.

The role of attention in this model is to select which item is sampled
at each time step. Thus, we view the problem of how to allocate
attention in decision making as a form of information search or active
learning. Importantly, the agent cannot simply decide to allocate
attention to the highest value item because she does not know the true
values. Rather, she must decide which item to attend to based on her
current estimations of each item's value. She may choose, for example,
to focus her attention on items that appear to have high value, or
perhaps items whose value is still highly uncertain. However, we do not
specify how expected value and uncertainty should be traded off (in
contrast to e.g.~upper confidence bound algorithms). Rather, we assume
that the agent allocates attention in an (approximately) optimal way.
Optimality here is defined as maximizing the quality of the final
decision minus the cost incurred by the decison making process.

In addition to deciding what information to gather (i.e.~what item to
attend to), the agent must decide when to stop gathering information. We
assume that this decision is also made optimally: The agent stops stops
gathering information and makes a decision when the expected increase in
decision quality falls below the expected cost of gathering more
information. As the sampling process progresses, the agent becomes
increasingly confident in her estimates, and as a result, the impact
(and thus potential benefit) of each new sample diminishes. In some
cases, however, the agent quickly identifies a choice with very high
value, and does not wait to attain high certainty in the exact value of
each item before making a choice.

Our model is based on theoretical work in artificial intelligence,
specifically the field of metareasoning \citep{Hay2012}, where provably
optimal sampling strategies have been identified for a narrow set of
problems. It is thus interesting, if not surprising, that the model has
much in common with the evidence accumulation models more familiar to
psychologists, in particular the drift diffusion model (DDM). As in the
DDM, decisions in our model arise from a process in which evidence for
each item is accumulated over time, being tracked by internal ``decision
variables'' that represent the estimated value of each item. However,
our model contrasts with the DDM in that both expected values and
uncertainty estimates are explicitly represented. This allows the
decision variables to evolve according to optimal Bayesian
inference.\footnote{Note the exception of DDM for binary hypothesis
  testing, which does implement optimal inference.} Furthermore, we do
not posit an explicit stopping rule such as a threshold, but instead use
an expected value computation to determine when to terminate the
decision making procedure. Finally, unlike any evidence accumulation
models of which we are aware, we propose that the information is not
gathered uniformly for all items, but is instead selected by a
near-optimal controller.

Having provided some intuition, we now present a formal model of
attention allocation for value-based choice. We model the problem as a
\emph{metalevel} Markov decision process, and discuss how we can find a
near-optimal solution to the problem using a specially designed
reinforcement learning algorithm.

\subsection{Metalevel Markov decision process
model}\label{metalevel-markov-decision-process-model}

A metalevel Markov decision process (meta-MDP) is a formalism developed
in the artificial intelligence literature to describe the problem of how
to allocate computational resources to best trade off between decision
quality and computational cost. The key insight lies in viewing
computation as a sequential process; an algorithm typicially executes
many individual operations to accomplish an ultimate goal, and the value
of each of those operations cannot be determined without reference to
all (or at least some) of the other operations in the sequence. Given
this observation, it is natural to model computation as a Markov
decision process because it is the standard formalism for modeling
sequential decision problems.

A meta-MDP is formally identical to a standard MDP, in that it is
defined by a set of states, a set of actions, a transition function, and
a reward function. It is distinct from a standard MDP only in its
interpretation and the way in which it is derived. In a meta-MDP the
states correspond to the agent's beliefs, the actions correspond to
computations (or cognitive operations), the transition function
describes how computations update the agent's beliefs, and the reward
function describes both the cost of computation and also the utility of
the item that is ultimately chosen.\footnote{Readers familiar with the
  Partially observable Markov decision process (POMDP) literature will
  observe that the metalevel MDP is similar to the belief-MDP
  representation of a POMDP, with the addition of replacing physical
  actions with a single operation that takes the optimal choice given
  the current belief.} We now define each of these elements.

The agent's beliefs are described by a set of Gaussian distributions,
each of which is the agent's posterior distribution of the utility of
one item. We denote the posterior utility distribution for item \(i\) at
time point \(t\) as
\(U_i(t) \sim \Normal(\mu_i(t), \lambda_i(t)^{-1})\). The belief can
thus be encoded by two vectors giving the mean and precision of the
estimate of each item's value. Thus, the state space of the meta-MDP is
\(\R^k \times (0,\infty)^k\).

The actions of the meta-MDP correspond to the computations (or cognitive
operations) that the agent can perform. Although many different kinds of
computations are likely to play a role in even simple decsions, we
consider a highly reduced set with one computation for each item,
\(\{c_1, c_2, \dots c_k \}\). Each computation draws a sample from an
obesrvation distribution for a single item. This distribution has mean
equal to the item's true utility \(u_i\) and some known variance
\(\sigma^2\). The posterior distribution for that item's utility \(U_i\)
is then updated according to Bayesian inference. In expectation, this
brings the MAP utility estimate (i.e. \(\mu_i\)) closer to the true
utility \(u_i\); however, due to sampling noise, a single computation
may actually increase the difference between the estimate and the ground
truth. This operation can be interpreted agnostically as ``considering
item \(i\)''.\footnote{Maybe connect to Bayesian brain sampling
  theories.} Additionally, we define a special computation, \(\bot\),
which indicates that the agent terminates the decision making process
and makes the best choice given her current beliefs, choosing an item
according to \(\arg\max_i \mu_i(t)\).

The metalevel transition function describes how computations affect
beliefs. We assume that the transition function is goverened by Bayesian
inference. Specifically, the posterior distribution of the item
considered at each time step is updated according to a sample drawn from
the corresponding item's observation distribution. Let \(c(t)\) denote
the item considered at time step \(t\). The transition dynamics are then
defined by the following equations:

\[
\begin{aligned}
o(t) &\sim \Normal(u_{c(t)}, \sigma^2) \\
\lambda_f(t+1) &= \lambda_f(t) + \sigma^{-2}  \\
\mu_f(t+1) &= \frac{\sigma^{-2} o(t) + \lambda_f(t) \mu_f(t)}{\lambda_f(t+1)}  \\
\lambda_i(t+1) &= \lambda_i(t) \text{ for } i \neq f  \\
\mu_i(t+1) &= \mu_i(t) \text{ for } i \neq f  \\
\end{aligned}
.
\]

The metalevel reward function captures both the costs of computation and
also the quality of the decision that is ultimately made. The costs are
given by the equation

\[
R(B_t, C_t) = -{\text{cost}_\text{sample}}- \mathbf{1}(C_t \neq C_{t-1}) {\text{cost}_\text{switch}},\]

where the first term captures the cost of sampling and the second
captures an additional switching cost for considering a different item
than the one considered on the previous time step. To maintain the
Markov property, we simply add the previous computation \(C_{t-1}\) as
an auxiliary state varibale. \({\text{cost}_\text{sample}}\) and
\({\text{cost}_\text{switch}}\) are free parameters of the model that
are fit to human data.

In addition to computational cost, the metalevel reward function also
captures decision quality by assigning a reward to the termination
action \(\bot\). When this action is selected, the agent chooses the
best item given her current beliefs, \(i^* = \arg\max_i \mu_i(t)\).
Thus, the most obvious way to define the reward for terminating would be
\(R(B_t, \bot) = u_{i^*}\). While this is a perfectly valid choice, we
take advantage of the fact that beliefs are unbiased in this model and
reduce the variance of the reward signal by marginalizing over possible
true utilities conditional on the agent's own beliefs. This results in

\[
R(B_t, \bot) = \max_i \mu_i(t)
,
\]

which has the same expectation as the previous defintion but lower
variance. This reduced variance faciliatates learning a metalevel policy
(as discussed below), and is the standard approach taken in the
metareasoning literature \citep{Hay2012}.

Having formalized the problem of attention allocation for decision
making as a metalevel Markov decision process, we now turn to our
solution which defines the optimal attention allocation policy. The
policy is defined in the usual way as a function that returns an action
(or distribution over actions) to take in a given state, the actions
being computations and the states being beliefs. The optimal policy is
the one that maximizes metalevel reward, or equivalently, the
\emph{value of computation}, which gives the expected benefit of
executing additional computations rather than making a decision
immediately. Formally, \(\text{VOC}(b, c)\) is defined as the expected
sum of all future metalevel rewards minus the reward of terminating in
belief state \(b\), given that computation \(c\) is executed in belief
state \(b\) and assuming that all future computations are chosen
optimally. Note that this function is nearly equivalent to the
state-action value function (often denoted \(Q\)) of a standard MDP,
\(\text{VOC}(b, c) = Q(b, c) - R(b, \bot)\).

Because the belife state space is continuous, standard dynamic
programming or tabular reinforcement learning techniques cannot be
applied; function approximation methods are necessary. We employ a
recently developed approximation method developed specifically for
meta-MDPS \citep{callaway2018learning}. The method uses hand-designed
\emph{value of information} features that give the value of executing a
single computation, the value of acquiring perfect knowledge about one
item, and the value of acquiring perfect knowledege about all items. The
first feature is lower bound on the VOC and the third is an upper bound.
The VOC is then approximated as a convex combination of these features,
with an additional term to capture expected future costs. We optimize
the weights of this combination to maximize the expected total metalevel
reward that the policy will receive on a random decision
problem.\footnote{Add more detail here. See Callaway et al. (2018) for
  details in the mean time.}

\subsection{Predictions}\label{predictions}

The model (comprising both the problem and optimal solution) makes
several qualitative predictions about the relationships between
attention, value, and choice.

\subsubsection{Value biases attention}\label{value-biases-attention}

In our model, the agent is more likely to attend to items that she
believes are valuable. To see why this is rational, consider a case in
which there are two items with high and similar value and one item with
much lower value. The agent can quickly determine that the low value
item is not the best one, and thus does not waste any additional time
considering it; thus, the two high-value items receive more attention.
In the reverse case, in which there is only one high value item, the
agent quickly identifies it, and has no reason to determine which of the
similarly low valued items is superior; attention is roughly equally
divided. Thus, in net, the positively valued items receive more
attention.

Importantly, the true value of items does not directly influence
attention; this effect is mediated by the internal value estimates that
the agent builds up over the course of a decision. In the initial stages
of a decision, the agent's value estimates will be highly uncertain and
noisy, with a weak dependence on the true values. Thus, although she
will tend to attend to the item she \emph{believes} is most valuable, it
is likely that this item is not truly the most valuable. As the decision
progresses, these estimates will become progressively less noisy. Thus,
the estimated values that are biasing attention will more closely align
with the true values. As a result, we expect to see that the tendency to
attend to high value items should increase over the course of the
decision.

Note that the attentional bias only holds for decisions between three or
more items. Indeed, it has been shown that attention should be exactly
evenly divided in the two alternative case \citep{Fudenberg2018} and our
model shows this behavior. Accordingly, no effect of value on attention
has been found in two-alternative choice experiments {[}citations{]}.

\subsubsection{Attention biases choice}\label{attention-biases-choice}

In our model, the more an agent attends to an item, the more likely she
is to ultimately choose it. This relationship holds even if we hold the
true values constant. Thus, we predict that people will be more likely
to choose an item that they looked at more, accounting for any
difference in the ratings previously assigned to each item. This
prediction is also made by the aDDM, a direct result of the assumption
that the drift rate is positively biased towards the attended item.
However, our model makes the same prediction without any assumption of
biased sampling.

There are two mechanisms through which the effect can emerge in our
model. The first mechanism is through a mismatch between the prior
distribution and the true distribution from which items are drawn. If
the mean of the true distribution is higher than the mean of the prior,
then sampling from an item will on average increase its estimated value.
Thus, as in the aDDM, the prediction arises from a causal (positive)
effect of attention on choice, mediated by estimated value.

The second mechanism by which the apparent attentional bias may emerge
depends on the rational allocation of attention. Under this explanation,
there is no positive effect of attention on value. Instead, there is a
positive effect of value on attention. That is, the rational model
predicts that (in the case of three or more items), the agent will be
more likely to attend to an item that she believes is valuable. At the
same time, she is also more likely to choose items that she believes are
valuable. Importantly, both choice and attention allocation depend on
\emph{estimated} value, not the true item values; thus, the causal
dependence is not broken by fixing (conditioning on) true value.

\begin{verbatim}
                  Possible figure

Mechanism #1: attention -> estimated value -> choice

Mechanism #2: attention <- estimated value -> choice
\end{verbatim}
ocess and integrate information in the service of making a decision. -->



<!-- One possible limitation  -->


<!-- One possible source of such a limitation on infromation processing is the "bottleneck" of attention [@Broadbent1958].  -->


<!-- A likely candidate for such a cognitive resource is attention. .. TODO ..    -->

A recent line of work has already begun to explore the role of (visual) attention in value-based decision making. The primary empirical effect is that people are more likely to choose items that they look at more. To explain this effect, @Krajbich2010 proposed the attentional drift diffusion model (aDDM), which biases the drift rate in favor of the fixated item. In addition to explaining the basic effect of attention on choice, the aDDM predicts subtler patterns. For example, last fixations are predicted to be shorter than first fixations, because the fixation is cut off when the decision variable crosses a barrier.

Despite these successes, several important questions have not been addressed by previous work. The vanilla drift-diffusion model has been shown to be equivalent to a sequential probability ratio test [@Bogacz2006; @Bitzer2014], providing a rational explanation for why people might use a DDM-like decision mechanism. However, there is no such rational justification for why people would bias their choices with their attention. Indeed, it is hard to imagine how such a bias could possibly improve the quality of choices. Nevertheless, there is some indication that people direct their attention in an adaptive manner: When presented with three or more options, people tend to look more at higher valued items, which in turn makes them more likely to choose those items. Although there is a clear intuition for why people would direct their attention in this manner, there is no formal model of how people allocate their attention in such cases, much less a rational model of how people _should_ allocate their attention.

We attempt to provide answers to both these questions (i.e. why people are more likely to choose items they attend to, and why people allocate their attention towards valuable items) with a single rational model. We begin by defining a Bayesian formulation of the drift-diffusion model, a process of sequential sampling and posterior updating. By endogenizing attention within this model (i.e. allowing an agent to select which item to consider at each time point), we define a sequential decision problem _for how to make a single decision_. We formalize this problem as _metalevel_ Markov decision process, and identify an approximately optimal solution using a recently developed reinforcement learning technique [@callaway2018learning]. Thus, we can make precise predictions about both how attention should affect choice as well as how attention should be allocated, all within a single rational model.

We apply the model to the dataset of Krajbich & Rangel (2010), in which participants repeatedly selected one out of three possible snacks, each of which was rated in a previous phase of the experiment. At choice time, visual attention was measures by eye tracking. Participants were given as much time as they like to make each choice. This rich dataset allows us to evaluate many different predictions of the model, including how attention affects choice and how attention develops as a function of both item value and previous fixations.

# Computational model

Before formally describing the model, we first provide some intuition. We treat decision making as an iterative process of sampling and inference. The decision maker (or _agent_) is presented with a set of items, each of which has some true unknown value. In order to determine which item to choose, the agent can generate noisy samples of each item's utility, each sample providing a small amount of information about the utility of a single item, but also having a small cost. The agent continously integrates information from these samples (along with a general prior over item value), thus developing an increasingly precise and accurate belief about each item's value.

The role of attention in this model is to select which item is sampled at each time step. Thus, we view the problem of how to allocate attention in decision making as a form of information search or active learning. Importantly, the agent cannot simply decide to allocate attention to the highest value item because she does not know the true values. Rather, she must decide which item to attend to based on her current estimations of each item's value. She may choose, for example, to focus her attention on items that appear to have high value, or perhaps items whose value is still highly uncertain. However, we do not specify how expected value and uncertainty should be traded off (in contrast to e.g. upper confidence bound algorithms). Rather, we assume that the agent allocates attention in an (approximately) optimal way. Optimality here is defined as maximizing the quality of the final decision minus the cost incurred by the decison making process.

In addition to deciding what information to gather (i.e. what item to attend to), the agent must decide when to stop gathering information. We assume that this decision is also made optimally: The agent stops stops gathering information and makes a decision when the expected increase in decision quality falls below the expected cost of gathering more information. As the sampling process progresses, the agent becomes increasingly confident in her estimates, and as a result, the impact (and thus potential benefit) of each new sample diminishes. In some cases, however, the agent quickly identifies a choice with very high value, and does not wait to attain high certainty in the exact value of each item before making a choice.

Our model is based on theoretical work in artificial intelligence, specifically the field of metareasoning [@Hay2012], where provably optimal sampling strategies have been identified for a narrow set of problems. It is thus interesting, if not surprising, that the model has much in common with the evidence accumulation models more familiar to psychologists, in particular the drift diffusion model (DDM). As in the DDM, decisions in our model arise from a process in which evidence for each item is accumulated over time, being tracked by internal "decision variables" that represent the estimated value of each item. However, our model contrasts with the DDM in that both expected values and uncertainty estimates are explicitly represented. This allows the decision variables to evolve according to optimal Bayesian inference.^[Note the exception of DDM for binary hypothesis testing, which does implement optimal inference.] Furthermore, we do not posit an explicit stopping rule such as a threshold, but instead use an expected value computation to determine when to terminate the decision making procedure. Finally, unlike any evidence accumulation models of which we are aware, we propose that the information is not gathered uniformly for all items, but is instead selected by a near-optimal controller.

<!-- Specifically, we assume that the decision maker (or _agent_) tracks Gaussian distributions over the value of each item. When first presented with a choice, each of these distributions is set to a common prior, capturing the distribution of item values across choice problems. As the decision making process progresses, samples are drawn from Gaussian distributions centered on the true (subjective) value of each item. After each sample, the distribution for the sampled item is updated by Bayesian posterior inference, integrating the prior (which may include information from previous samples) with the likelihood provided by each new sample. -->


Having provided some intuition, we now present a formal model of attention allocation for value-based choice. We model the problem as a _metalevel_ Markov decision process, and discuss how we can find a near-optimal solution to the problem using a specially designed reinforcement learning algorithm.

## Metalevel Markov decision process model
A metalevel Markov decision process (meta-MDP) is a formalism developed in the artificial intelligence literature to describe the problem of how to allocate computational resources to best trade off between decision quality and computational cost. The key insight lies in viewing computation as a sequential process; an algorithm typicially executes many individual operations to accomplish an ultimate goal, and the value of each of those operations cannot be determined without reference to all (or at least some) of the other operations in the sequence. Given this observation, it is natural to model computation as a Markov decision process because it is the standard formalism for modeling sequential decision problems.

A meta-MDP is formally identical to a standard MDP, in that it is defined by a set of states, a set of actions, a transition function, and a reward function. It is distinct from a standard MDP only in its interpretation and the way in which it is derived. In a meta-MDP the states correspond to the agent's beliefs, the actions correspond to computations (or cognitive operations), the transition function describes how computations update the agent's beliefs, and the reward function describes both the cost of computation and also the utility of the item that is ultimately chosen.^[Readers familiar with the Partially observable Markov decision process (POMDP) literature will observe that the metalevel MDP is similar to the belief-MDP representation of a POMDP, with the addition of replacing physical actions with a single operation that takes the optimal choice given the current belief.] We now define each of these elements.

The agent's beliefs are described by a set of Gaussian distributions, each of which is the agent's posterior distribution of the utility of one item. We denote the posterior utility distribution for item $i$ at time point $t$ as $U_i(t) \sim \Normal(\mu_i(t), \lambda_i(t)^{-1})$. The belief can thus be encoded by two vectors giving the mean and precision of the estimate of each item's value. Thus, the state space of the meta-MDP is $\R^k \times (0,\infty)^k$.

The actions of the meta-MDP correspond to the computations (or cognitive operations) that the agent can perform. Although many different kinds of computations are likely to play a role in even simple decsions, we consider a highly reduced set with one computation for each item, $\{c_1, c_2, \dots c_k \}$. Each computation draws a sample from an obesrvation distribution for a single item. This distribution has mean equal to the item's true utility $u_i$ and some known variance $\sigma^2$. The posterior distribution for that item's utility $U_i$ is then updated according to Bayesian inference. In expectation, this brings the MAP utility estimate (i.e. $\mu_i$) closer to the true utility $u_i$; however, due to sampling noise, a single computation may actually increase the difference between the estimate and the ground truth. This operation can be interpreted agnostically as "considering item $i$".^[Maybe connect to Bayesian brain sampling theories.] Additionally, we define a special computation, $\bot$, which indicates that the agent terminates the decision making process and makes the best choice given her current beliefs, choosing an item according to $\arg\max_i \mu_i(t)$.

The metalevel transition function describes how computations affect beliefs. We assume that the transition function is goverened by Bayesian inference. Specifically, the posterior distribution of the item considered at each time step is updated according to a sample drawn from the corresponding item's observation distribution. Let $c(t)$ denote the item considered at time step $t$. The transition dynamics are then defined by the following equations:

$$
\begin{aligned}
o(t) &\sim \Normal(u_{c(t)}, \sigma^2) \\
\lambda_f(t+1) &= \lambda_f(t) + \sigma^{-2}  \\
\mu_f(t+1) &= \frac{\sigma^{-2} o(t) + \lambda_f(t) \mu_f(t)}{\lambda_f(t+1)}  \\
\lambda_i(t+1) &= \lambda_i(t) \text{ for } i \neq f  \\
\mu_i(t+1) &= \mu_i(t) \text{ for } i \neq f  \\
\end{aligned}
.
$$


The metalevel reward function captures both the costs of computation and also the quality of the decision that is ultimately made. The costs are given by the equation

\newcommand{\samplecost}{\text{cost}_\text{sample}}
\newcommand{\switchcost}{\text{cost}_\text{switch}}
$$
R(B_t, C_t) = -\samplecost - \mathbf{1}(C_t \neq C_{t-1}) \switchcost
,$$

where the first term captures the cost of sampling and the second captures an additional switching cost for considering a different item than the one considered on the previous time step. To maintain the Markov property, we simply add the previous computation $C_{t-1}$ as an auxiliary state varibale. $\samplecost$ and $\switchcost$ are free parameters of the model that are fit to human data.

In addition to computational cost, the metalevel reward function also captures decision quality by assigning a reward to the termination action $\bot$. When this action is selected, the agent chooses the best item given her current beliefs, $i^* = \arg\max_i \mu_i(t)$. Thus, the most obvious way to define the reward for terminating would be $R(B_t, \bot) = u_{i^*}$. While this is a perfectly valid choice, we take advantage of the fact that beliefs are unbiased in this model and reduce the variance of the reward signal by marginalizing over possible true utilities conditional on the agent's own beliefs. This results in

$$
R(B_t, \bot) = \max_i \mu_i(t)
,
$$

which has the same expectation as the previous defintion but lower variance. This reduced variance faciliatates learning a metalevel policy (as discussed below), and is the standard approach taken in the metareasoning literature [@Hay2012].

Having formalized the problem of attention allocation for decision making as a metalevel Markov decision process, we now turn to our solution which defines the optimal attention allocation policy. The policy is defined in the usual way as a function that returns an action (or distribution over actions) to take in a given state, the actions being computations and the states being beliefs. The optimal policy is the one that maximizes metalevel reward, or equivalently, the _value of computation_, which gives the expected benefit of executing additional computations rather than making a decision immediately. Formally, $\text{VOC}(b, c)$ is defined as the expected sum of all future metalevel rewards minus the reward of terminating in belief state $b$, given that computation $c$ is executed in belief state $b$ and assuming that all future computations are chosen optimally. Note that this function is nearly equivalent to the state-action value function (often denoted $Q$) of a standard MDP, $\text{VOC}(b, c) = Q(b, c) - R(b, \bot)$.

Because the belife state space is continuous, standard dynamic programming or tabular reinforcement learning techniques cannot be applied; function approximation methods are necessary. We employ a recently developed approximation method developed specifically for meta-MDPS [@callaway2018learning]. The method uses hand-designed _value of information_ features that give the value of executing a single computation, the value of acquiring perfect knowledge about one item, and the value of acquiring perfect knowledege about all items. The first feature is lower bound on the VOC and the third is an upper bound. The VOC is then approximated as a convex combination of these features, with an additional term to capture expected future costs. We optimize the weights of this combination to maximize the expected total metalevel reward that the policy will receive on a random decision problem.^[Add more detail here. See Callaway et al. (2018) for details in the mean time.]


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