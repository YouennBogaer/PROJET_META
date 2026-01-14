import numpy as np
import time
import os
import argparse


from src.python.script.MOP import MOP
from src.python.script.obj_functions import obj_functions
from src.python.script.MOAED import MOEAD
from src.python.script.params import params
from src.python.script.weigth import generate_weights_simplex
from src.python.script.save_results import enregistrer_json, afficher_pareto_3d

N_tasks = params['N']
tasks_data = {
    'G': np.random.randint(10, 100, N_tasks),
    'RG': np.random.randint(1, 20, N_tasks),
    'parent': np.random.randint(1, params['M']+1, N_tasks)
}



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="resultat", help="name of resuslts")
    parser.add_argument("--stop", type=int, default=100, help="ncriterion stop number of génération")
    parser.add_argument("--save_json", action="store_true", help="Sauvegarder les résultats en JSON")
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
    my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * N_tasks, bounds_x=bornes)

    weigths = generate_weights_simplex(m=3, H=20)

    moead = MOEAD(my_mop, stop_criterion=args.stop, weights=weigths, len_neighborhood=15)

    moead.execute()
    end = time.time()
    print(f"time : {end-start}")

    if not os.path.exists("results"):
        os.makedirs("results")
        print("Results directory created")
    afficher_pareto_3d(moead,filename)

    if args.save_json:
        enregistrer_json(moead, filename=filename)