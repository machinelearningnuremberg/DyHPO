import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('white')

from benchmarks.lcbench import LCBench
from benchmarks.taskset import TaskSet

from hpo_method import DyHPOAlgorithm


parser = argparse.ArgumentParser(
    description='DyHPO experiments.',
)
parser.add_argument(
    '--index',
    type=int,
    default=1,
)
parser.add_argument(
    '--fantasize_step',
    type=int,
    default=1,
)
parser.add_argument(
    '--budget_limit',
    type=int,
    default=1000,
)
parser.add_argument(
    '--dataset_name',
    type=str,
    default='covertype',
)
parser.add_argument(
    '--benchmark_name',
    type=str,
    default='lcbench',
)
parser.add_argument(
    '--project_dir',
    type=str,
    default='C:\\Users\\arlin\\Desktop\\DyHPO', #'/home/arlind/Desktop/DyHPO',
)
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output',
)

args = parser.parse_args()
benchmark_name = args.benchmark_name
dataset_name = args.dataset_name
fantasize_step = args.fantasize_step
budget_limit = args.budget_limit

if benchmark_name == 'lcbench':
    benchmark_extension = os.path.join(
        'lc_bench',
        'results',
        'data_2k.json',
    )
elif benchmark_name == 'taskset':
    benchmark_extension = os.path.join(
        'data',
        'taskset',
    )
else:
    raise NotImplementedError('At the current version of the code only two benchmarks are supported')

benchmark_data_path = os.path.join(
    args.project_dir,
    benchmark_extension,
)

seeds = np.arange(10)
seed = seeds[args.index - 1]
random.seed(seed)

output_dir = os.path.join(
    args.output_dir,
    f'{benchmark_name}',
    'dyhpo',
)
os.makedirs(output_dir, exist_ok=True)

surrogate_types = {
    'lcbench': LCBench,
    'taskset': TaskSet,
}

minimization_problem_type = {
    'lcbench': False,
    'taskset': True,
}

minimization = minimization_problem_type[benchmark_name]

benchmark = surrogate_types[benchmark_name](benchmark_data_path, dataset_name)

min_value = 0
max_value = 0

if benchmark_name == 'taskset':
    max_value = benchmark.max_value
    min_value = benchmark.min_value

dyhpo_surrogate = DyHPOAlgorithm(
    benchmark.get_hyperparameter_candidates(),
    benchmark.log_indicator,
    seed=seed,
    max_benchmark_epochs=benchmark.max_budget,
    fantasize_step=fantasize_step,
    minimization=minimization,
    total_budget=budget_limit,
    dataset_name=dataset_name,
    output_path=output_dir,
)

evaluated_configs = dict()
method_budget = 0
incumbent = 0
method_trajectory = []
dyhpo_budgets = []

while method_budget < budget_limit:

    hp_index, budget = dyhpo_surrogate.suggest()
    performance_curve = benchmark.get_curve(hp_index, budget)
    score = performance_curve[-1]
    dyhpo_surrogate.observe(hp_index, budget, performance_curve)
    budget_cost = 0
    if hp_index in evaluated_configs:
        previous_budget = evaluated_configs[hp_index]
        budget_cost = budget - previous_budget
        evaluated_configs[hp_index] = budget
    else:
        budget_cost = fantasize_step

    method_budget += budget_cost

    if score > incumbent:
        incumbent = score
        method_trajectory.append(incumbent)
        dyhpo_budgets.append(method_budget)

random_budget = 0
np.random.seed(seed)

random_trajectory = []
random_budgets = []
incumbent = 0
while random_budget < budget_limit:

    config_index = np.random.randint(0, 2000)
    performance = benchmark.get_performance(config_index, benchmark.max_budget)
    random_budget += benchmark.max_budget
    if performance > incumbent:
        incumbent = performance
        random_trajectory.append(incumbent)
        random_budgets.append(random_budget)

benchmark_incumbent_value = max(benchmark.get_incumbent_curve())
regret_incumbent_trajectory_dyhpo = [benchmark_incumbent_value - inc_performance for inc_performance in method_trajectory]
regret_incumbent_trajectory_random = [benchmark_incumbent_value - inc_performance for inc_performance in random_trajectory]

# duplicating the last entry at the end of the budget so both curves can be
# of the same length
regret_incumbent_trajectory_dyhpo.append(regret_incumbent_trajectory_dyhpo[-1])
regret_incumbent_trajectory_random.append(regret_incumbent_trajectory_random[-1])

# adding the last budget to the curve
dyhpo_budgets.append(1000)
random_budgets.append(1000)

plt.plot(dyhpo_budgets, regret_incumbent_trajectory_dyhpo, label='DyHPO')
plt.plot(random_budgets, regret_incumbent_trajectory_random, label='Random')
plt.yscale('log')
plt.legend()
plt.tight_layout()

plt.savefig('example.pdf')
