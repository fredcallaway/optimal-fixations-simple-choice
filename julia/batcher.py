#!/usr/bin/env python3
import itertools
import os
import click
import subprocess

def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(itertools.product(*d.values())):
        yield dict(zip(d.keys(), v))

SBATCH_SCRIPT = '''
#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=runs/{job_name}/out/%A_%a
#SBATCH --array=1-{n_job}
#SBATCH --time={max_time}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mail-type=end
#SBATCH --mail-user=flc2@princeton.edu

module load julia
julia -p {cpus_per_task} optimize.jl {job_name} $SLURM_ARRAY_TASK_ID
'''.strip()


def params(quick):
    return dict_product({
        'n_arm': 3,
        'n_iter': 10 if quick else 100,
        'n_roll': 100 if quick else 1000,
        'obs_sigma': [3,5,7],
        'sample_cost': [0.002, 0.001],
        'switch_cost': [4, 8, 12],
        'seed': [0]
})

from scipy.stats import uniform
def rand_params(quick):
    for i in range(1000):
        yield {
            'n_arm': 3,
            'n_iter': 10 if quick else 100,
            'n_roll': 100 if quick else 1000,
            'obs_sigma': uniform(1, 20).rvs().round(3),
            'sample_cost': uniform(.001, .009).rvs().round(6),
            'switch_cost': uniform(1, 14).rvs().round(3),
            'seed': 0
        }

@click.command()
@click.argument('job-name')
@click.argument('max-time')
@click.option('--quick', is_flag=True)
@click.option('--mem-per-cpu', default=5000)
@click.option('--cpus-per-task', default=1)
def main(job_name, quick, **slurm_args):
    os.makedirs(f'runs/{job_name}/jobs', exist_ok=True)
    os.makedirs(f'runs/{job_name}/out', exist_ok=True)
    import json
    for i, prm in enumerate(rand_params(quick), start=1):
        prm['group'] = job_name
        with open(f'runs/{job_name}/jobs/{i}.json', 'w+') as f:
            json.dump(prm, f)

    with open('run.sbatch', 'w+') as f:
        f.write(SBATCH_SCRIPT.format(n_job=i, job_name=job_name, **slurm_args))

    print(f'Wrote JSON and run.sbatch with {i} jobs.')
    subprocess.Popen('sbatch --test-only run.sbatch', shell=True)


if __name__ == '__main__':
    main()
