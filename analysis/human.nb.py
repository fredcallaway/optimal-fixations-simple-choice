# ---
# jupyter:
#   anaconda-cloud: {}
#   jupytext_format_version: '1.2'
#   jupytext_formats: ipynb,nb.py
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.3
#   nav_menu: {}
#   toc:
#     navigate_menu: true
#     number_sections: true
#     sideBar: true
#     threshold: 6
#     toc_cell: false
#     toc_section_display: block
#     toc_window_display: false
# ---

# +
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context('notebook', font_scale=1.3)
sns.set_palette('tab10')
sns.set_style('white')
# -

from toolz import concat
df = pd.read_csv('../krajbich_PNAS_2011/data.csv')
df['fixation'] = list(concat(df.groupby(['subject', 'trial']).apply(lambda x: range(len(x)))))
df.drop('Unnamed: 0', axis=1, inplace=True)
df.set_index(['subject', 'trial'], inplace=True)
df.rename(columns={
    'roirating': 'roi_rating',
    'leftroi': 'roi1',
    'middleroi': 'roi2',
    'rightroi': 'roi3'
}, inplace=True)
subjects = df.reset_index().subject.unique()

trial_ratings = df.groupby(['subject', 'trial'])[[f'rating{i}' for i in range(1,4)]].mean()
trial_ratings.mean(axis=1)

# +
trial_ratings = df.groupby(['subject', 'trial'])[[f'rating{i}' for i in range(1,4)]].mean()
rating_agg = (pd.melt(trial_ratings.reset_index(), id_vars=['subject', 'trial']) 
              .groupby(['subject']).value.agg(['mean', 'std']))

value = trial_ratings.copy()
for s in subjects:
    value.loc[s] = ((value.loc[s] - rating_agg['mean'].loc[s]) / rating_agg['std'].loc[s]).values
# -

import h5py
hf = h5py.File('../data.h5', 'w')
hf.create_dataset('values', data=value.values)
hf.close()

roi = df[['roi1', 'roi2', 'roi3']].as_matrix()
df['roi'] = (roi * [1,2,3]).sum(1)
choice = df[['choice1', 'choice2', 'choice3']].as_matrix()
df['choice'] = (choice * [1,2,3]).sum(1)
rank = df[['rating1', 'rating2', 'rating3']].rank(1, ascending=False).astype(int).as_matrix()
roi_rank = (roi * rank).sum(1)
df['roi_rank'] = roi_rank

# +
def pull_values(row, key):
    x = row[[f'{key}{i}' for i in (1,2,3)]].copy()
    x.index = [1,2,3]
    return x

def others(i):
    return {1,2,3} - {i}

trial = df.loc[1, 1]
def parse_trial(trial):
    last = trial.iloc[-1].astype(int)
    rating = pull_values(last, 'rating')
    fixtime = trial.groupby('roi').eventduration.sum()
    t = {
        'rt': last.rt,
        'subject': last.subject,
        'trial': last.trial,
        'last_fix': last.roi,
        'choice': last.choice,
        'last_duration': last.eventduration,
#         'choose_last': last.choice == last.roi,
#         'last_fix_value_advantage': rating[last.roi] - rating[others(last.roi)].mean()
#         'last_value': last[f'rating{last.roi}']
    }
#     for i in range(1,4):
#         t[f'rating{i}'] = last[f'rating{i}']
#         t[f'fixtime{i}'] = fixtime.get(i, 0)
    return t

parse_trial(trial.reset_index())
# tdf = pd.DataFrame(list(df.reset_index().groupby(['subject', 'trial']).apply(parse_trial))).set_index(['subject', 'trial'])
# -

tdf['n_fix'] = df.groupby(['subject', 'trial']).apply(len)
tdf['value_std'] = value.values.std(1)
tdf['value_max'] = value.values.max(1)
tdf['value_mean'] = value.values.mean(1)
tdf['choice_value'] = value.values[np.arange(len(value)), tdf.choice.values - 1]
tdf.reset_index().to_pickle('data/human_trials')

# ## Choice quality

choice = np.array(tdf.choice - 1)
tdf['choice_value'] = ratings_z.values[np.arange(len(ratings_z)), choice]
tdf.choice_value.agg(['mean', 'std'])

# # Attention over time

# +
def long_form(df, quant=50):
    for (sub, t), trial in df.reset_index().groupby(['subject', 'trial']):
        step = 0
        for _, fix in (trial.iterrows()):
            for _ in range(int(round(fix.eventduration / quant))):
                x = fix[['subject', 'trial', 'roi', 'roi_rating_z', 'roi_rank']]
                x['step'] = step
                step += quant
                yield x

max_rt = tdf.rt.quantile(0.99)
ldf = pd.DataFrame(long_form(df.reset_index().query('rt < @max_rt')))
# -

ldf[['step', 'roi_rank']].rename(columns={'step': 'time', 'roi_rank': 'focus_rank'}).to_pickle('data/human_focus')

# +
def plot_fixations(ldf):
    n_step = ldf.groupby(['roi_rank', 'step']).apply(len)
    p_step = n_step / n_step.sum(level=1)
    p_step.unstack().T.plot()
    plt.xlim(1,3000)
    plt.ylim(0, 1)
    plt.ylabel('Probability of Fixating')
    plt.xlabel('Time')
    plt.legend(title='Item Value Rank')
    plt.axhline(1/3, c='k', lw=1, ls='--')
    plt.title('Human Fixations');
#     plt.savefig('figs/human_fix.pdf')

plot_fixations(ldf)

# +
# s, t = np.random.choice(tdf.index)
fig, axes = plt.subplots(1,2, gridspec_kw = {'width_ratios':[3, 1]})
ldf.set_index(['subject', 'trial']).loc[s, t].roi.plot(ax=axes[0])
axes[0].axhline(tdf.loc[s, t].choice, c='r', ls='--')

ratings_z.loc[s, t].plot.barh(ax=axes[1])
axes[1].set_yticks(())

