import numpy as np
import time
import os
import argparse

from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex,
                               enregistrer_json,
                               afficher_pareto_3d
                               )


N_tasks = params['N']
tasks_data = {
    'G': np.random.randint(10, 100, N_tasks),
    'RG': np.random.randint(1, 20, N_tasks),
    'parent': np.random.randint(1, params['M'] + 1, N_tasks)
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="resultat", help="name of resuslts")
    parser.add_argument("--stop", type=int, default=100, help="ncriterion stop number of génération")
    parser.add_argument("--save_json", action="store_true", help="Sauvegarder les résultats en JSON")
    parser.add_argument("--mutation", type=float, default=0.2, help="Define the percentage of mutation for each child")
    parser.add_argument("--H", type=int, default=10,
                        help="Definit le nombre de découpage que l'on fait de notre problème")
    parser.add_argument("--neighbor", type=int, default=5,
                        help="Definit le nombre de voisisns des vecteurs à qui on va comparer le nouvel enfant")
    parser.add_argument("--evaluate", action="store_true", help="Compare our implementation of MOEAD with another project")

    #################################################################################################
    #
    #   Our implementation of the MOEA/D algorithm for
    #   "Gestion des ressources cloud : optimisation de l’allocation entre serveurs MEC et cloud"
    #   multiobjectif problem.
    #
    #################################################################################################
    # On conseil de visiter les voisins entre 5 et 10% du nombre de dimension on obtient le nombre de dimensions en
    # faisant m-1 parmis H et donc dans notre cas 3 parmi H

    # TODO : rajouter les metadonnés noter quelque part après une éxécution

    args = parser.parse_args()
    filename = args.name
    start = time.time()
    obj_functions_list = [
        lambda x: obj_functions(x, params, tasks_data)[0],  # F1 : Temps
        lambda x: obj_functions(x, params, tasks_data)[1],  # F2 : Énergie
        lambda x: obj_functions(x, params, tasks_data)[2]  # F3 : Charge
    ]

    bornes = [(0, params['M']), (1, 7)] * N_tasks
    # TODO Remarque : pk dim_x = 2 * N_tasks et pas juste 2 ?
    # Pcq un chromosome à une forme [srvtask 1,cputask 1,srv2,cpu2 ..., srvNtask,cpuNtasks] on a plusieurs tass en parallèle
    my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * N_tasks, bounds_x=bornes)

    H = args.H
    weigths = generate_weights_simplex(m=3, H=H)

    moead = MOEAD(my_mop,
                  stop_criterion=args.stop,
                  weights=weigths,
                  len_neighborhood=args.neighbor,
                  params=params,
                  percentage_mutation=args.mutation)

    moead.execute()
    end = time.time()
    print(f"time : {end - start}")

    if not os.path.exists("results"):
        os.makedirs("results")
        print("Results directory created")
    afficher_pareto_3d(moead, filename)

    if args.save_json:
        enregistrer_json(moead, filename=filename)

    #################################################################################################
    #
    #   Our implementation of the MOEA/D algorithm for
    #   "Gestion des ressources cloud : optimisation de l’allocation entre serveurs MEC et cloud"
    #   multiobjectif problem.
    #
    #################################################################################################
    if args.evaluate:
        print("Yo")