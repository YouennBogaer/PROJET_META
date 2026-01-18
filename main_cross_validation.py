import csv
import os
import time
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from src.python.script.run import run
from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex,
                               enregistrer_json,
                               afficher_pareto_3d,
                               )

from src.python.script.metrics import calculate_spacing, hypervolume, pf


def execute_config(args):
    """ Exécuté par un worker pour tester une configuration spécifique """
    id_conf, stop, h_val, mut, neigh, N_runs, tasks_data = args

    # Calcul du nombre de sous-problèmes et du voisinage
    sub_problem = int(((h_val + 2) * (h_val + 1)) / 2)
    nnb_neigh = int(sub_problem * neigh)

    config = {
        'mutation': mut,
        'neighbor': nnb_neigh,
        'H': h_val,
        'stop': stop
    }

    start = time.time()
    hvs, spaces, nb_solutions = run(N_runs, config, tasks_data)
    duration = (time.time() - start) / N_runs

    # Préparation des lignes pour le CSV
    rows = []
    for i in range(len(hvs)):
        rows.append([id_conf, i, mut, neigh, h_val, stop, nb_solutions[i], hvs[i], spaces[i], duration])

    return rows


if __name__ == '__main__':
    # Paramètres du Grid Search
    N_runs = 5
    mutations = [0.1, 0.2, 0.4, 0.5]
    neighbors = [0.05, 0.1, 0.15, 0.20]
    criterion = [100, 200, 500, 1000]
    H_list = [8, 10, 15, 25]

    # Génération des données de test
    tasks_data = {
        'G': np.random.randint(10, 100, params['N']),
        'RG': np.random.randint(1, 20, params['N']),
        'parent': np.random.randint(1, params['M'] + 1, params['N'])
    }

    total_configs = len(mutations) * len(neighbors) * len(criterion) * len(H_list)
    print(f"Nombre de configurations : {total_configs}")
    print(f"Nombre total d'exécutions : {total_configs * N_runs}")

    # Création de la liste des combinaisons à tester
    config_id = 1
    tasks_to_process = []
    for stop in criterion:
        for h_val in H_list:
            for mut in mutations:
                for neigh in neighbors:
                    tasks_to_process.append((config_id, stop, h_val, mut, neigh, N_runs, tasks_data))
                    config_id += 1

    file_save = "results_parameters/grid_search.csv"
    if not os.path.exists("results_parameters"):
        os.makedirs("results_parameters")

    # Gestion du pool multi-processus
    num_cores = mp.cpu_count() - 1
    print(f"Lancement sur {num_cores} cœurs...")

    start_total = time.time()
    all_results = []
    with mp.Pool(processes=num_cores) as pool:
        # Suivi de la progression avec tqdm
        for result in tqdm(pool.imap_unordered(execute_config, tasks_to_process), total=len(tasks_to_process)):
            all_results.append(result)

    # Exportation finale en CSV
    with open(file_save, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id_conf', 'run_id', 'mutation', 'neighbor', 'H', 'stop', 'nb solutions', 'hv', 'spacing',
                         'duration_batch'])
        for config_rows in all_results:
            writer.writerows(config_rows)

    print(f"Temps total : {time.time() - start_total:.2f}s")
    print(f"Résultats sauvegardés dans : {file_save}")