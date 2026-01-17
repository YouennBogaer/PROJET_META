from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex,
                               enregistrer_json,
                               afficher_pareto_3d,
                               )
from src.python.script.save_results import plot_parallel_coordinates
from src.python.script.metrics import calculate_spacing, hypervolume,pf


def run(n_runs, current_params, tasks_data):
    """Lance plusieurs fois le même algo pour avoir une moyenne robuste grrr"""
    hv_results = []
    spacing_results = []
    nb_solution = []
    # --- POINT DE RÉFÉRENCE FIXE ---
    # On le fixe ici pour que TOUS les tests utilisent la même base de comparaison
    # Adapte ces valeurs selon les limites max réelles de ton problème
    fixed_ref_point = [25.0, 150000.0, 1.1]

    for r in range(n_runs):
        #print(f"  Run {r + 1}/{n_runs}...", end="\r")

        obj_functions_list = [
            lambda x: obj_functions(x, params, tasks_data)[0],
            lambda x: obj_functions(x, params, tasks_data)[1],
            lambda x: obj_functions(x, params, tasks_data)[2]
        ]

        bornes = [(0, params['M']), (1, 7)] * (2 * params['N'])  # Ajusté selon ta logique
        my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * params['N'], bounds_x=bornes)

        weights = generate_weights_simplex(m=3, H=current_params['H'])

        moead = MOEAD(my_mop,
                      stop_criterion=current_params['stop'],
                      weights=weights,
                      len_neighborhood=current_params['neighbor'],
                      params=params,
                      percentage_mutation=current_params['mutation'])

        moead.execute()

        # Calcul des métriques
        current_pf = pf(moead)
        s = calculate_spacing(current_pf)
        # On passe le point de référence fixe à ta future fonction HV
        h = hypervolume(current_pf, ref_point=fixed_ref_point)

        hv_results.append(h)
        spacing_results.append(s)
        nb_solution.append(len(current_pf))
    return hv_results, spacing_results, nb_solution