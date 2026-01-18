import numpy as np
import time
import os
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex)

from src.python.script.metrics import calculate_spacing, hypervolume, pf
SCENARIOS_TASKS = [40, 280]
# Scénarios basés sur les noms des fichiers excel disponibles
SCENARIOS_TASKS = [40, 80, 120, 160, 200, 240, 280]



def get_best_solution_topsis(pareto_front, weights=[1 / 3, 1 / 3, 1 / 3]):
    """
    Sélectionne la meilleure option via la méthode TOPSIS
    """
    matrice_decision = np.array(pareto_front)

    # Normalisation de la matrice
    denominateurs = np.sqrt(np.sum(matrice_decision ** 2, axis=0))
    denominateurs[denominateurs == 0] = 1
    matrice_norm = matrice_decision / denominateurs

    # Application des poids
    matrice_ponderee = matrice_norm * weights

    # Identification des solutions idéales et anti-idéales (minimisation)
    ideal_solution = np.min(matrice_ponderee, axis=0)
    anti_ideal_solution = np.max(matrice_ponderee, axis=0)

    # Calcul des distances euclidiennes
    dist_pos = np.sqrt(np.sum((matrice_ponderee - ideal_solution) ** 2, axis=1))
    dist_neg = np.sqrt(np.sum((matrice_ponderee - anti_ideal_solution) ** 2, axis=1))

    # Calcul du score de proximité relative
    closeness = dist_neg / (dist_pos + dist_neg)
    return matrice_decision[np.argmax(closeness)]


def run_single_experiment(seed, args, n_tasks, param_template):
    """
    Exécution d'une instance unique de l'algorithme MOEA/D
    """
    np.random.seed(seed)

    # Copie locale des paramètres pour éviter les conflits en multiprocessing
    current_params = param_template.copy()

    # Chargement des données réelles depuis les différentes feuilles du fichier Excel
    filename = f"data/task{n_tasks}.xlsx"
    try:
        # Lecture des tâches
        df_t = pd.read_excel(filename, sheet_name='TaskDetails')
        # Lecture des nœuds pour automatiser M
        df_n = pd.read_excel(filename, sheet_name='NodeDetails')

        # Mise à jour dynamique des dimensions
        current_n = len(df_t)
        current_m = len(df_n)

        current_params['N'] = current_n
        current_params['M'] = current_m

        local_tasks_data = {
            'G': df_t['Number of instructions (109 instructions)'].astype(float).values,
            'RG': df_t['Output file size (MB)'].astype(float).values,
            # Le parent est assigné aléatoirement parmi les serveurs MEC disponibles (1 à M)
            'parent': np.random.randint(1, current_m + 1, current_n)
        }
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")
        return None

    # Définition des fonctions objectifs : Temps, Énergie, Équilibrage de charge
    obj_functions_list = [
        lambda x: obj_functions(x, current_params, local_tasks_data)[0],
        lambda x: obj_functions(x, current_params, local_tasks_data)[1],
        lambda x: obj_functions(x, current_params, local_tasks_data)[2]
    ]

    # Définition des bornes de recherche (Serveur ID, CPU alloué)
    max_cpu = current_params['mnv']
    bornes = [(0, current_m), (1, max_cpu)] * current_n

    # Initialisation du problème multi-objectif
    my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * current_n, bounds_x=bornes, params=current_params)

    # Configuration de MOEA/D
    H = args.H
    sub_problem = int(((H + 2) * (H + 1)) / 2)
    nnb_neigh = int(sub_problem * args.neighbor)
    weights = generate_weights_simplex(m=3, H=H)

    moead = MOEAD(my_mop, stop_criterion=args.stop, weights=weights,
                  len_neighborhood=nnb_neigh, params=current_params, percentage_mutation=args.mutation)

    # Exécution de l'algorithme
    _, _ = moead.execute()
    pareto_front = pf(moead)

    if len(pareto_front) == 0:
        return None

    # Sélection de la meilleure solution sur le front de Pareto via TOPSIS
    best_sol = get_best_solution_topsis(pareto_front)
    return {"f1": best_sol[0], "f2": best_sol[1], "f3": best_sol[2]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="Nombre de runs par scénario")
    parser.add_argument("--stop", type=int, default=200, help="Nombre de générations")
    parser.add_argument("--mutation", type=float, default=0.3, help="Taux de mutation")
    parser.add_argument("--H", type=int, default=8, help="Paramètre de division pour les poids")
    parser.add_argument("--neighbor", type=float, default=0.05, help="Taille du voisinage en pourcentage")
    args = parser.parse_args()

    results_f1, results_f2, results_f3 = [], [], []

    print(f"Début de la simulation : {args.runs} runs par fichier")

    # Utilisation de n-2 coeurs pour préserver la stabilité du système
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Calcul parallèle activé sur {num_workers} cœurs")
    start_global = time.time()

    if not os.path.exists("results_comparaison_plot"):
        os.makedirs("results_comparaison_plot")

    # Itération sur les fichiers de benchmark
    for n_tasks in SCENARIOS_TASKS:
        print(f"Traitement de l'instance task{n_tasks}.xlsx...")
        seeds = [np.random.randint(0, 10000) for _ in range(args.runs)]

        # Préparation de la fonction de travail pour le pool
        worker_func = partial(run_single_experiment, args=args, n_tasks=n_tasks, param_template=params)

        # Exécution parallèle des runs pour le scénario actuel
        with multiprocessing.Pool(processes=num_workers) as pool:
            raw_results = pool.map(worker_func, seeds)

        # Filtrage et calcul des moyennes des indicateurs de performance
        valid_results = [r for r in raw_results if r is not None]
        if valid_results:
            results_f1.append(np.mean([r['f1'] for r in valid_results]))
            results_f2.append(np.mean([r['f2'] for r in valid_results]))
            results_f3.append(np.mean([r['f3'] for r in valid_results]))
        else:
            results_f1.append(0)
            results_f2.append(0)
            results_f3.append(0)

    print(f"Simulation terminée en {time.time() - start_global:.2f} secondes")

    # Génération des graphiques de synthèse
    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f1, width=20, color='royalblue', label='MOEA/D', alpha=0.8)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Completion time (ms)')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('results_comparaison_plot/completion_time.png')

    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f3, width=20, color='seagreen', label='MOEA/D', alpha=0.8)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Load balance variance')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('results_comparaison_plot/load_balance.png')

    plt.figure(figsize=(8, 5))
    plt.bar(SCENARIOS_TASKS, results_f2, width=20, color='indianred', label='MOEA/D', alpha=0.8)
    plt.xlabel('Number of IoT applications')
    plt.ylabel('Total energy consumption')
    plt.xticks(SCENARIOS_TASKS)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('results_comparaison_plot/energy_consumption.png')

    print("Les graphiques ont été exportés dans le dossier results_comparaison_plot")