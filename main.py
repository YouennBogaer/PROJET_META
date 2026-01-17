import numpy as np
import time
import os
import argparse

from matplotlib import pyplot as plt

from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex,
                               enregistrer_json,
                               afficher_pareto_3d,
                               )

from src.python.script.save_results import plot_parallel_coordinates
from src.python.script.metrics import calculate_spacing, hypervolume, pf


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
    parser.add_argument("--neighbor", type=float, default=5,
                        help="Definit le nombre de voisisns des vecteurs à qui on va comparer le nouvel enfant")

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

    max_cpu = params['mnv']
    bornes = [(0, params['M']), (1, max_cpu)] * N_tasks
    # TODO Remarque : pk dim_x = 2 * N_tasks et pas juste 2 ?
    # Pcq un chromosome à une forme [srvtask 1,cputask 1,srv2,cpu2 ..., srvNtask,cpuNtasks] on a plusieurs tass en parallèle
    my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * N_tasks, bounds_x=bornes,params=params)

    H = args.H
    neigh = args.neighbor
    sub_problem = int(((H + 2) * (H + 1)) / 2)
    nnb_neigh = int(sub_problem * neigh)

    print(f"nb voisins : {nnb_neigh}")
    weigths = generate_weights_simplex(m=3, H=H)
    print(f"nb weigths : {len(weigths)}")
    print(f"% voisins {(nnb_neigh/len(weigths))*100}")
    moead = MOEAD(my_mop,
                  stop_criterion=args.stop,
                  weights=weigths,
                  len_neighborhood=nnb_neigh,
                  params=params,
                  percentage_mutation=args.mutation)

    list_EP,list_HP = moead.execute()
    end = time.time()
    print(f"time : {end - start}")

    if not os.path.exists("results"):
        os.makedirs("results")
        print("Results directory created")


    if args.save_json:
        enregistrer_json(moead, filename=filename)

    pf = pf(moead)
    print(f"taille du front en point : {len(pf)}")
    spacing = calculate_spacing(pf)
    print(f"spacing : {spacing}")

    fixed_ref_point = [20.0, 65000.0, 20.0]
    hv = hypervolume(pf, fixed_ref_point)
    print(f"hypervolume : {hv}")
    afficher_pareto_3d(moead, filename)
    plot_parallel_coordinates(pf, filename)

    plt.plot(list_EP)

    plt.savefig("results/" + "EP_list")
    plt.show()
    plt.plot(list_HP)
    plt.savefig("results/" + "HPList")

    plt.show()

    #################################################################################################
    #
    #   Comparison with another implementation
    #
    #################################################################################################
    if args.evaluate:
        print("Yo")