import numpy as np
import time
import argparse
import os
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt
from src.python.script import (params, obj_functions)
from src.python.script.metrics import calculate_spacing
from src.python.script.save_results import afficher_pareto

# Scénarios basés sur les noms des fichiers excel disponibles
SCENARIOS_TASKS = [40, 80, 120, 160, 200, 240, 280]


class MyOptimizationProblem:
    def __init__(self, n_tasks, params_dict, tasks_data):
        self.n_tasks = n_tasks
        self.params = params_dict
        self.tasks_data = tasks_data
        # Bornes : serveur [0, M] et CPU [1, mnv]
        self.low_bounds = [0.0, 1.0] * n_tasks
        self.high_bounds = [float(params_dict['M']), float(params_dict['mnv'])] * n_tasks

    def repair(self, y):
        y = np.array(y)
        M = self.params['M']
        cap_max = self.params['mnv']

        for i in range(len(y)):
            if i % 2 == 0:
                y[i] = int(np.clip(y[i], 0, M))
            else:
                y[i] = int(np.clip(y[i], 1, cap_max))

        usage = np.zeros(M + 1)
        tasks_overflowing = []

        for n in range(self.n_tasks):
            srv = int(y[2 * n])
            cpu = int(y[2 * n + 1])
            if srv > 0:
                if usage[srv] + cpu <= cap_max:
                    usage[srv] += cpu
                else:
                    tasks_overflowing.append(n)

        for n in tasks_overflowing:
            cpu = int(y[2 * n + 1])
            target_idx = np.argmin(usage[1:])
            target_srv = target_idx + 1

            if usage[target_srv] + cpu <= cap_max:
                y[2 * n] = target_srv
                usage[target_srv] += cpu
            else:
                y[2 * n] = 0
        return y

    def fitness(self, x):
        x_repaired = self.repair(x.copy())
        res = obj_functions(x_repaired, self.params, self.tasks_data)
        dist_penalty = np.mean(np.square(x - x_repaired)) * 1e-6
        return [res[0] + dist_penalty, res[1] + dist_penalty, res[2] + dist_penalty]

    def get_bounds(self):
        return (self.low_bounds, self.high_bounds)

    def get_nobj(self):
        return 3


def run_nsga2_experiment(n_tasks, args, global_params):
    filename = f"data/task{n_tasks}.xlsx"
    try:
        df_t = pd.read_excel(filename, sheet_name='TaskDetails')
        df_n = pd.read_excel(filename, sheet_name='NodeDetails')

        local_params = global_params.copy()
        local_params['N'] = len(df_t)
        local_params['M'] = len(df_n)

        local_tasks_data = {
            'G': df_t['Number of instructions (109 instructions)'].astype(float).values,
            'RG': df_t['Output file size (MB)'].astype(float).values,
            'parent': np.random.randint(1, local_params['M'] + 1, local_params['N'])
        }
    except Exception as e:
        print(f"Erreur sur le fichier {filename}: {e}")
        return None

    prob = pg.problem(MyOptimizationProblem(local_params['N'], local_params, local_tasks_data))
    algo = pg.algorithm(pg.nsga2(gen=args.stop, cr=0.8, eta_c=20, m=0.1, eta_m=20))
    pop = pg.population(prob, size=args.pop_size)
    pop = algo.evolve(pop)

    fits = pop.get_f()
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    pareto_front = fits[ndf[0]]

    return {
        "f1": np.mean(pareto_front[:, 0]),
        "f2": np.mean(pareto_front[:, 1]),
        "f3": np.mean(pareto_front[:, 2])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=int, default=500, help="nombre de générations")
    parser.add_argument("--pop_size", type=int, default=100, help="taille population")
    args = parser.parse_args()

    results_f1, results_f2, results_f3 = [], [], []

    print("Lancement de la simulation NSGA-II sur les instances réelles")
    start_global = time.time()

    for n_tasks in SCENARIOS_TASKS:
        print(f"Calcul NSGA-II pour l'instance {n_tasks} tâches...")
        res = run_nsga2_experiment(n_tasks, args, params)

        if res:
            results_f1.append(res['f1'])
            results_f2.append(res['f2'])
            results_f3.append(res['f3'])
        else:
            results_f1.append(0);
            results_f2.append(0);
            results_f3.append(0)

    os.makedirs("results_nsga2_plots", exist_ok=True)

    # Graphique Temps de complétion
    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f1, width=20, color='royalblue', label='NSGA-II', alpha=0.8)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Completion time (ms)')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('results_nsga2_plots/nsga2_completion_time.png')

    # Graphique Variance de la charge
    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f3, width=20, color='seagreen', label='NSGA-II', alpha=0.8)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Load balance variance')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('results_nsga2_plots/nsga2_load_balance.png')

    # Graphique Consommation Énergie
    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f2, width=20, color='indianred', label='NSGA-II', alpha=0.8)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Total energy consumption')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('results_nsga2_plots/nsga2_energy_consumption.png')

    print(f"Simulation terminée en {time.time() - start_global:.2f}s")
    print("Tous les graphiques sont dans le dossier results_nsga2_plots")