import numpy as np
import time
import os
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from functools import partial

from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex)

from src.python.script.metrics import calculate_spacing, hypervolume, pf

# nb tasks
SCENARIOS_TASKS = [50, 100, 150, 200, 250]


def get_best_solution_topsis(pareto_front, weights=[1 / 3, 1 / 3, 1 / 3]):
    """
    Sélectionne la meilleure option via TOPSIS
    """
    matrice_decision = np.array(pareto_front)

    # Normalisation
    denominateurs = np.sqrt(np.sum(matrice_decision ** 2, axis=0))
    denominateurs[denominateurs == 0] = 1
    matrice_norm = matrice_decision / denominateurs

    # Pondération
    matrice_ponderee = matrice_norm * weights

    # Recherche des idéaux (on veut minimiser les 3 objectifs)
    ideal_solution = np.min(matrice_ponderee, axis=0)
    anti_ideal_solution = np.max(matrice_ponderee, axis=0)

    # Calcul des distances
    dist_pos = np.sqrt(np.sum((matrice_ponderee - ideal_solution) ** 2, axis=1))
    dist_neg = np.sqrt(np.sum((matrice_ponderee - anti_ideal_solution) ** 2, axis=1))

    # Score final
    closeness = dist_neg / (dist_pos + dist_neg)
    return matrice_decision[np.argmax(closeness)]


def run_single_experiment(seed, args, n_tasks,param):
    """
    Un seul run de l'algo
    """
    np.random.seed(seed)

    # Setup des data de tâches
    local_tasks_data = {
        'G': np.random.randint(10, 100, n_tasks),
        'RG': np.random.randint(1, 20, n_tasks),
        'parent': np.random.randint(1, param['M'] + 1, n_tasks)
    }

    # Fonctions : Temps, Énergie, Charge
    obj_functions_list = [
        lambda x: obj_functions(x, param, local_tasks_data)[0],
        lambda x: obj_functions(x, param, local_tasks_data)[1],
        lambda x: obj_functions(x, param, local_tasks_data)[2]
    ]

    # Init du problème
    max_cpu = param['mnv']
    bornes = [(0, param['M']), (1, max_cpu)] * n_tasks
    my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * n_tasks, bounds_x=bornes, params=param)

    # Setup MOEA/D
    H = args.H
    sub_problem = int(((H + 2) * (H + 1)) / 2)
    nnb_neigh = int(sub_problem * args.neighbor)
    weights = generate_weights_simplex(m=3, H=H)

    moead = MOEAD(my_mop, stop_criterion=args.stop, weights=weights,
                  len_neighborhood=nnb_neigh, params=param, percentage_mutation=args.mutation)

    # Lancement
    moead.execute()
    pareto_front = pf(moead)

    if len(pareto_front) == 0:
        return None

    # On récupère le meilleur point
    best_sol = get_best_solution_topsis(pareto_front)
    return {"f1": best_sol[0], "f2": best_sol[1], "f3": best_sol[2]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Nb de tests par pack")
    parser.add_argument("--stop", type=int, default=1500, help="Nb générations")
    parser.add_argument("--mutation", type=float, default=0.25, help="Taux mutation")
    parser.add_argument("--H", type=int, default=15, help="Découpage poids")
    parser.add_argument("--neighbor", type=float, default=0.2, help="Taille voisinage %")
    parser.add_argument("--tasks", type=int, default=50)  # Pour compatibilité
    args = parser.parse_args()

    # Data pour les plots
    results_f1, results_f2, results_f3 = [], [], []

    print(f"Lancement de la simu : {args.runs} runs par scenario")

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    #J'ai mis sur n-2 car n-1 a fait planter mon ordi et abimé mon ssd apparemment mais je sais pas si ça marche sur windows
    print(f"Tavail en cour sur {num_workers} couers")
    start_global = time.time()

    # On fait défiler les scénarios un par un
    for n_tasks in SCENARIOS_TASKS:
        print(f"Calcul pour {n_tasks} tâches...")
        seeds = [np.random.randint(0, 10000) for _ in range(args.runs)]
        params['N'] = n_tasks
        worker_func = partial(run_single_experiment, args=args, n_tasks=n_tasks, param=params)

        # Parallélisation des runs d'un même scénario
        with multiprocessing.Pool(processes=num_workers) as pool:
            raw_results = pool.map(worker_func, seeds)

        # On vire les None et on fait la moyenne
        valid_results = [r for r in raw_results if r is not None]
        results_f1.append(np.mean([r['f1'] for r in valid_results]))
        results_f2.append(np.mean([r['f2'] for r in valid_results]))
        results_f3.append(np.mean([r['f3'] for r in valid_results]))

    print(f"Terminé en {time.time() - start_global:.2f}s")

    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f1, width=30, color='b', label='MOEA/D', alpha=0.7)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Completion time (ms)')
    plt.xticks(SCENARIOS_TASKS)  # Pour que les nombres soient bien sous les barres
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('completion_time.png')

    # Charge
    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f3, width=30, color='g', label='MOEA/D', alpha=0.7)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Load balance variance')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('load_balance.png')

    # Énergie
    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f2, width=30, color='r', label='MOE/D', alpha=0.7)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Total energy consumption')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('energy_consumption.png')

    print("Fichiers sauvegardés : completion_time.png, load_balance.png, energy_consumption.png")
