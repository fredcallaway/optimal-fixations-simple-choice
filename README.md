# Code and data for "Fixation patterns in simple choice reflect optimal information sampling"

Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008863


If you would like to use part of this code for your own project, I encourage you to get in touch with me before you dive into the code (it's a bit hairy) fredcallaway@princeton.edu

## Important files

- `meta_mdp.jl` defines the metalevel Markov decision process.
- `voi.jl` defines the value of information features.
- `bmps.jl` defines the policy, based on the VOI features.
- `bmps_ucb.jl` defines the UCB method for identifying near-optimal settings of the BMPS weights.
- `pseudo_likelihood.jl` defines the likelihood of summary statistics used to fit the model.

## Model simulations

Simulations from the optimal model can be found [here](https://drive.google.com/drive/folders/1h6mpdTnKbyBe4hX93_ofowCmA0bGjPVW?usp=share_link)

That folder contains all data needed to plot the model simulations in the paper. Each file contains the simulations for binary or trinary choice, with the prior either "fit" to the data, fixed to the empirical mean ("unbiased") or fixed to "zero".

The files are JSON formatted, a list of trial objects. The spec for each trial is:

- `value`: a list specifying the value of the two/three options
- `choice`: the chosen option, **ONE-INDEXED** (1-2 or 1-3)
- `fixations`: a list specifying which option was fixated at each time step (100ms), also one-indexed
- `param_idx`: (1-30) id for the parameter configuration
- `policy_idx`: (1-80) id for the near-optimal policy

The last two values can be ignored in most cases. Together they roughly correspond to one "subject"

## Fitting steps

I don't imagine anyone else will want to invest the compute resources to actually replicate this, but here are instructions to do so for the sake of completeness.

- Define parameters such as the search space and output directory in `fit_base.jl`. I use files in `runs/` to organize parameters for different runs. The final run used to generate results in the paper is `runs/revision.jl`.
- Find the near-optimal policies for each candidate parameter setting: run `julia compute_policies.jl i` for i in 1 to N. Each job takes an average of 30 minutes.
- Compute the likelihood for each candidate parameter setting: run `julia compute_likelihood.jl i` for i in 1 to N. Each job takes an average of 64 minutes. Each job depends only on the results of the same index from the compute_policies step.
- Identify the top-performing parmaeters based on the results from the previous step: Run `julia process_likelihood.jl`. Six sets are identified in which the prior is either fit or not fit and the fitting is done jointly or separately on each dataset.
- Compute likelihood and simulations on the held out set: Run `julia evaluation.jl joint PRIOR_MODE`. PRIOR_MODE can be `fit`, `unbiased`, or `zero`. To generate the full results, you need to run all three modes.
- Plots are generated in `plots.jl` and `individual_plots.jl`. This must be done in an environment where graphics can be displayed. I used the Hydrogen package in Atom. Alternatively, you can use X11 forwarding to run it on a server, or you can copy the simulation results to your local machine (warning: each set of simulations is 500-750 MB).
- The ADDM is implemented in `addm.jl`. Generate simulations and precompute features to plot with `julia run_addm.jl`. `plot_addm_binary.jl` and `plot_addm_trinary.jl` generate the replication plots to compare against the original papers.
<!-- - Run `julia plots.jl RUN_NAME FIT_MODE-FIT_PRIOR`.  As such, it is  -->

