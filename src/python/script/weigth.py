import numpy as np
from itertools import combinations_with_replacement


def generate_weights_simplex(m, H):
    """
    Génère des vecteurs de poids uniformément répartis.
    m : nombre d'objectifs (ici 3)
    H : nombre de divisions (définit la densité)
    """
    # Le nombre total de vecteurs sera C(H+m-1, m-1)
    weights = []
    # On génère toutes les combinaisons possibles
    for combo in combinations_with_replacement(range(H + 1), m - 1):
        # On calcule les coordonnées sur le simplexe
        # print(f" combo {combo}")
        # print(f" combo type  {type (combo)}")

        # On ajoute le début et la fin
        points = [0] + sorted(list(combo)) + [H]
        w = []
        for i in range(len(points) - 1):
            # On normalise les poids entre 0 et 1 (en gros on disvise juste par H et en prenant les points de 2 poids
            # on obtient une valeurs entre O et 1  et la somme de tous les poids donnent 1
            w.append((points[i + 1] - points[i]) / H)
        weights.append(np.array(w))
    print(f"number of sub-problems generated {len(weights)}")
    return weights


# test
if __name__ == '__main__':
    # Avec H=12 et m=3, on obtient 91 vecteurs de poids
    mes_poids = generate_weights_simplex(m=3, H=12)
    print(f"Nombre de sous-problèmes générés : {len(mes_poids)}")

