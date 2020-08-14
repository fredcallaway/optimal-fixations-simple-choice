
## Important files

- `meta_mdp.jl` defines the metalevel Markov decision process.
- `voi.jl` defines the value of information features.
- `bmps.jl` defines the policy, based on the VOI features.
- `bmps_ucb.jl` defines the UCB method for identifying near-optimal settings of the BMPS weights.
- `pseudo_likelihood.jl` defines the likelihood of summary statistics used to fit the model.

## Fitting steps

I don't imagine anyone else will want to invest the compute resources to actually replicate this, but here are instructions to do so for the sake of completeness. Note that the code allows for fitting the prior mean, but we don't do so for the main results because doing so yields inconclusive results.

- Define parameters such as the search space and output directory in `fit_base.jl`. I use files in `runs/` to organize parameters for different runs.
- Find the near-optimal policies for each candidate parameter setting: run `julia compute_policies.jl i` for i in 1 to N. Each job takes an average of 30 minutes.
- Compute the likelihood for each candidate parameter setting: run `julia compute_likelihood.jl i` for i in 1 to N. Each job takes an average of 64 minutes. Each job depends only on the results of the same index from the compute_policies step.
- Identify the top-performing parmaeters based on the results from the previous step: Run `julia identify_mle.jl`. Four sets are identified in which the prior is either fit or not fit and the fitting is done jointly or separately on each dataset.
- Compute likelihood and simulations on the held out set: Run `julia evaluation.jl FIT_MODE FIT_PRIOR` for each desired combination. `FIT_MODE` can be either `joint` or `separate` and `FIT_PRIOR` can be either `true` or `false`.
- Plots are generated in `plots.jl`. This must be done in an environment where graphics can be displayed. I used the Hydrogen package in Atom. Alternatively, you can use X11 forwarding to run it on a server, or you can copy the simulation results to your local machine (warning: each set of simulations is 500-750 MB).
<!-- - Run `julia plots.jl RUN_NAME FIT_MODE-FIT_PRIOR`.  As such, it is  -->



