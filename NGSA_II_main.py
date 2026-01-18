import numpy as np
import time
import argparse
import pygmo as pg

# Imports locaux
from src.python.script import (params, obj_functions)
from src.python.script.metrics import calculate_spacing, hypervolume
from src.python.script.save_results import afficher_pareto

# Setup des données de base
N_tasks = params['N']
tasks_data = {
    'G': np.random.randint(10, 100, N_tasks),
    'RG': np.random.randint(1, 20, N_tasks),
    'parent': np.random.randint(1, params['M'] + 1, N_tasks)
}


class MyOptimizationProblem:
    def __init__(self, n_tasks, params, tasks_data):
        self.n_tasks = n_tasks
        self.params = params
        self.tasks_data = tasks_data
        # Bornes par défaut : serveur et CPU
        self.bounds_x = [[0, params['M']], [1, 7]] * n_tasks

    def repair(self, y):
        # On remet les valeurs dans les clous
        y = np.array(y)
        for i in range(len(y)):
            low, high = self.bounds_x[i]
            y[i] = int(np.clip(y[i], low, high))

        M = self.params['M']
        cap_max = self.params['mnv']
        usage = np.zeros(M + 1)

        tasks_overflowing = []
        for n in range(len(y) // 2):
            srv = int(y[2 * n])
            cpu = int(y[2 * n + 1])
            if srv > 0:
                if usage[srv] + cpu <= cap_max:
                    usage[srv] += cpu
                else:
                    tasks_overflowing.append(n)

        # Réassignation des tâches qui débordent
        for n in tasks_overflowing:
            cpu = int(y[2 * n + 1])
            target_idx = np.argmin(usage[1:])
            target_srv = target_idx + 1

            if usage[target_srv] + cpu <= cap_max:
                y[2 * n] = target_srv
                usage[target_srv] += cpu
            else:
                y[2 * n] = 0  # Envoi vers le Cloud
        return y

    def fitness(self, x):
        # On répare avant d'évaluer
        x_repaired = self.repair(x.copy())

        # Calcul des 3 objectifs
        res = obj_functions(x_repaired, self.params, self.tasks_data)

        # Pénalité minime pour garder la trace du changement
        dist_penalty = np.mean(np.square(x - x_repaired)) * 1e-6

        return [res[0] + dist_penalty, res[1] + dist_penalty, res[2] + dist_penalty]

    def get_bounds(self):
        # Bornes pour Pygmo
        low = [0.0, 1.0] * self.n_tasks
        high = [float(self.params['M']), 7.0] * self.n_tasks
        return (low, high)

    def get_nobj(self):
        # Nombre d'objectifs à minimiser
        return 3



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="res_pygmo_nsga2", help="nom des resultats")
    parser.add_argument("--stop", type=int, default=1000, help="nombre de générations")
    args = parser.parse_args()

    # Création du problème
    prob = pg.problem(MyOptimizationProblem(N_tasks, params, tasks_data))

    # Configuration de l'algo NSGA-II
    algo = pg.algorithm(pg.nsga2(
        gen=args.stop,
        cr=0.8,
        eta_c=20,
        m=0.5,
        eta_m=20
    ))

    # Génération de la population
    pop = pg.population(prob, size=500)

    start = time.time()

    pop = algo.evolve(pop)
    end = time.time()

    print(f"Temps d'exécution : {end - start:.4f}s")

    # Extraction du Front de Pareto
    fits = pop.get_f()
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    pareto_front = fits[ndf[0]]

    # Graphique final
    afficher_pareto(pop, args.name)

    if len(pareto_front) > 0:
        #  Point de référence pour la normalisation
        fixed_ref_point = np.array([5000.0, 500000.0, 50000.0])
        f_min = np.array([0.0, 0.0, 0.0])


        pf_norm = (pareto_front - f_min) / (fixed_ref_point - f_min)

        # Calcul du Spacing
        spacing = calculate_spacing(pareto_front)

        #  Calcul de l'Hypervolume normalisé
        # On utilise [1.1, 1.1, 1.1] comme référence sur le front normalisé
        ref_norm = [1.1, 1.1, 1.1]
        hv_obj = pg.hypervolume(pf_norm)
        hv_value = hv_obj.compute(ref_norm)

        # Score final entre 0 et 1
        hv_final = hv_value / (1.1 ** 3)

        print(f"Points sur le front : {len(pareto_front)}")
        print(f"Spacing : {spacing:.4f}")
        print(f"Hypervolume Normalisé : {hv_final:.4f}")

    else:
        print("Aucune solution trouvée..")
