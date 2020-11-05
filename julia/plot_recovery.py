import pandas as pd
from figures import Figures
figs = Figures('figs/recovery')
figs.watch()
show = figs.show

true = pd.read_csv('results/revision/recovery/results/true.csv')
top1 = pd.read_csv('results/revision/recovery/results/top1.csv')
top30 = pd.read_csv('results/revision/recovery/results/top30.csv')
top1_50 = pd.read_csv('results/revision/recovery/results/top1-50.csv')
top30_50 = pd.read_csv('results/revision/recovery/results/top30-50.csv')
keys = true.columns
# %% --------

# %% --------

lims = {
    "α": (100, 500),
    "σ_obs": (1, 5),
    "sample_cost": (.001, .01),
    "switch_cost": (.003, .03),
    "β_μ": (0, 1)
}
tex = {
    "α": r"$\beta$",
    "σ_obs": r"$\sigma_x$",
    "sample_cost": r"$\gamma_\mathrm{sample}$",
    "switch_cost": r"$\gamma_\mathrm{switch}$",
    "β_μ": r"$\alpha$",
}
texhat = {
    "α": r"$\hat{\beta}$",
    "σ_obs": r"$\hat{\sigma}_x$",
    "sample_cost": r"$\hat{\gamma}_\mathrm{sample}$",
    "switch_cost": r"$\hat{\gamma}_\mathrm{switch}$",
    "β_μ": r"$\hat{\alpha}$",
}

def set_lims(lo, hi):
    r = hi - lo
    lo -= .05 * r
    hi += .05 * r
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)

from scipy.stats import pearsonr

def plot_param_cor(n_item, method):
    fit = fits[n_item, method]
    plot_cor(fit)

def plot_cor(true, fit):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for (k, ax) in zip(keys, axes.flat):
        plt.sca(ax)
        lo, hi = lims[k]
        # plt.scatter(true[k], fit[k], c='k', s=10, alpha=0.5)
        sns.regplot(true[k], fit[k], 
            color='C0', scatter_kws=dict(color='k', s=10, alpha=0.1))
        plt.plot([lo, hi], [lo, hi], c='r', ls=':')
        set_lims(lo, hi)
        plt.xticks([lo, hi], [lo, hi])
        plt.yticks([lo, hi], [lo, hi])
        plt.xlabel(f'{tex[k]}')
        plt.ylabel(f'{texhat[k]}')

        # plt.axvline([true[k].mean()], alpha=0.8, ls=':')
        # plt.axhline([fit[k].mean()], alpha=0.8, ls=':')

        ax.set_aspect('equal', 'box')
        r = pearsonr(true[k], fit[k])[0]
        plt.annotate(f'$r = {r:.3f}$', (0.1, 0.9), xycoords='axes fraction', bbox=dict(facecolor='white', alpha=0.8))
    axes.flat[-1].axis('off')


plot_cor(true, top30); show('top30', pdf=True)

# %% --------
plot_cor(true, top30_50); show('top30_50', pdf=True)

plot_cor(true, top1); show('top1', pdf=True)
plot_cor(true, top1_50); show('top1_50', pdf=True)

# %% --------
sns.regplot(alt_true['β_μ'], alt_mle['β_μ'], 
    color='C0', scatter_kws=dict(color='k', s=10, alpha=0.5))
show()

# %% --------
def compute_bias(true, fit):
    rng = pd.Series([lims[k][1] - lims[k][0] for k in keys], index=keys)
    return (fit.mean() - true.mean()) / rng

compute_bias(true, top30)

# %% --------
# keep = alt_true.β_μ == 1
# plot_cor(alt_true.loc[keep], alt_mle.loc[keep])
# show()

# %% --------
plot_cor(true, jfit)
show()

# %% --------
    # plt.tight_layout()
    # plt.savefig(f'figs/recovery/cor-{n_item}-{method}.pdf')

# for n_item in [2, 3]:
#     for method in ['group', 'indiv']:
#         figs.figure(n_item=n_item, method=method)(plot_param_cor)

# %% ==================== Rank of true parameter ====================

import seaborn as sns
# rank.plot.hist()

fig, axes = plt.subplots(2,2)
axx = iter(axes.flat)
for n_item in [2, 3]:
    for method in ['group', 'indiv']:
        rank = ranks[n_item, method]
        plt.sca(next(axx))
        sns.distplot(rank, kde=False)
        plt.title(f'{n_item} {method}')

for ax in axes[1, :]:
    ax.set_xlabel('Rank')
for ax in axes[:, 0]:
    ax.set_ylabel('Count')
show()

# %% ==================== Correlation plots ====================
X = jfit.rename(columns=texhat)

sns.pairplot(X, kind='reg', 
    plot_kws=dict(scatter_kws=dict(alpha=0.2))
)
show()

