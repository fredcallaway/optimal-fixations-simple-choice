#!/usr/bin/env python3
import click
import pandas as pd
from glob import glob
import json
import numpy as np


def read_rollouts(job_name):
    for fn in glob(f'../julia/runs/{job_name}/results/rollouts-*'):
        with open(fn) as f: 
            j = json.load(f)
            for roll in j['rollouts']:
                roll['focused'] = np.array(roll['focused'])
                roll['value'] = np.array(roll['value'])
                for k, v in j['params'].items():
                    roll[k] = v
                yield roll



def n_fix(roll):
    return (np.diff(roll) != 0).sum() + 1

@click.group()
def cli(): pass

@cli.command()
@click.argument('jobname')
def model(jobname):
    rollouts = list(read_rollouts(jobname))
    df = pd.DataFrame(rollouts)

    df['n_step'] = df.focused.apply(len)
    df['n_fix'] = df.focused.apply(n_fix)

    value = np.stack(df.value)
    df['value_std'] = value.std(1)
    choice = np.array(df.choice - 1)
    df['choice_value'] = value[np.arange(len(value)), choice]
    df.to_pickle(f'data/{jobname}')


def human():
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
    # df['mean_rating'] = trial_ratings.mean(1)
    # df['min_rating'] = trial_ratings.min(1)
    # df['roi_relative_rating'] = df.roi_rating - df.mean_rating
    # SUBJECTS = df.reset_index().subject.unique()

if __name__ == '__main__':
    cli()