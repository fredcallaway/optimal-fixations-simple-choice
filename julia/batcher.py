#!/usr/bin/env python3
import random
from math import log, exp
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
julia -p {cpus_per_task} {file} {job_name} $SLURM_ARRAY_TASK_ID
'''.strip()

def write_bash_script(file, job_name, cpus_per_task, n_job):
    my_id = random.randint(0, 100000)
    with open('run.sh', 'w+') as f:
        f.write('#!/usr/bin/env bash\n')
        for i in range(n_job):
            f.write(f'julia -p {cpus_per_task} {file} {job_name} {i+1} '
                    f'&> runs/{job_name}/out/sh_{my_id}_{i+1} &\n')


def params(quick):
    return dict_product({
        'n_arm': 3,
        'n_iter': 10 if quick else 200,
        'n_roll': 100 if quick else 1000,
        'obs_sigma': [5],
        'sample_cost': [0.002, 0.004],
        'switch_cost': [4, 8],
        'seed': [1,2,3,4],
        'cost_features': [1,2,3]
})


def uniform(a, b):
    return a + random.random() * (b - a)

def log_uniform(a, b):
    return exp(uniform(log(a), log(b)))


log_uniform(0.001, 0.1)

def rand_params(quick):
    for i in range(1000):
        yield {
            'n_arm': 3,
            'n_iter': 10 if quick else 200,
            'n_roll': 100 if quick else 1000,
            'obs_sigma': uniform(1, 20),
            'sample_cost': log_uniform(.001, .1),
            'switch_cost': uniform(1, 20),
            'seed': 0,
        }

@click.command()
@click.argument('file')
@click.argument('job-name')
@click.argument('max-time')
@click.option('--quick', is_flag=True)
@click.option('--mem-per-cpu', default=5000)
@click.option('--cpus-per-task', default=1)
@click.option('--no-test', '-v', is_flag=True)
def main(file, job_name, quick, no_test, **slurm_args):
    os.makedirs(f'runs/{job_name}/jobs', exist_ok=True)
    os.makedirs(f'runs/{job_name}/out', exist_ok=True)
    import json
    for i, prm in enumerate(rand_params(quick), start=1):
        prm['group'] = job_name
        with open(f'runs/{job_name}/jobs/{i}.json', 'w+') as f:
            json.dump(prm, f)

    with open('run.sbatch', 'w+') as f:
        f.write(SBATCH_SCRIPT.format(n_job=i, file=file, job_name=job_name, **slurm_args))

    write_bash_script(file, job_name, slurm_args['cpus_per_task'], i)

    print(f'Wrote runs/{job_name}/jobs/*.json, run.sh, and run.sbatch with {i} jobs.')
    if not no_test:
        print('sbatch --test-only run.sbatch')
        subprocess.Popen('sbatch --test-only run.sbatch', shell=True)


if __name__ == '__main__':
    main()
