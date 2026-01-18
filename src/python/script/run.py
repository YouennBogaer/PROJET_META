from src.python.script import (MOP,
                               obj_functions,
                               MOEAD,
                               params,
                               generate_weights_simplex,
                               enregistrer_json,
                               afficher_pareto_3d,
                               )

from src.python.script.metrics import calculate_spacing, hypervolume, pf


def run(n_runs, current_params, tasks_data):
    """ Exécute l'algorithme plusieurs fois pour obtenir des moyennes robustes """
    hv_results = []
    spacing_results = []
    nb_solution = []

    # Référence fixe pour comparer l'hypervolume sur tous les tests
    fixed_ref_point = [25.0, 150000.0, 1.1]

    for r in range(n_runs):
        # Définition des 3 fonctions objectifs (Temps, Énergie, Charge)
        obj_functions_list = [
            lambda x: obj_functions(x, params, tasks_data)[0],
            lambda x: obj_functions(x, params, tasks_data)[1],
            lambda x: obj_functions(x, params, tasks_data)[2]
        ]

        # Bornes des variables de décision
        bornes = [(0, params['M']), (1, 7)] * (2 * params['N'])
        my_mop = MOP(obj_functions=obj_functions_list, dim_x=2 * params['N'], bounds_x=bornes)

        # Génération des vecteurs de poids pour MOEA/D
        weights = generate_weights_simplex(m=3, H=current_params['H'])

        # Initialisation de l'algorithme
        moead = MOEAD(my_mop,
                      stop_criterion=current_params['stop'],
                      weights=weights,
                      len_neighborhood=current_params['neighbor'],
                      params=params,
                      percentage_mutation=current_params['mutation'])

        # Lancement de l'optimisation
        moead.execute()

        # Récupération et calcul des performances du front de Pareto
        current_pf = pf(moead)
        s = calculate_spacing(current_pf)
        h = hypervolume(current_pf, ref_point=fixed_ref_point)

        hv_results.append(h)
        spacing_results.append(s)
        nb_solution.append(len(current_pf))

    return hv_results, spacing_results, nb_solution