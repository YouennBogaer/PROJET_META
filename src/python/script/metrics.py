from pymoo.indicators.hv import HV
import numpy as np

def pf(moead) :
    return np.array([pair[1] for pair in moead.ex_pop])


def hypervolume(pf,ref_point) :
    #max_found = np.max(pf, axis=0)
    #ref_point = np.maximum(max_found * 1.1, [600, 12000, 1.0])
    f_min = np.array([0.0, 0.0, 0.0])
    f_max = np.array(ref_point)


    pf_norm = (pf - f_min) / (f_max - f_min)


    ref_norm = np.array([1.1, 1.1, 1.1])
    ind = HV(ref_point=ref_norm)
    hv_value = ind(pf_norm)
    return hv_value/np.prod(ref_norm)


def calculate_spacing(front):

    if len(front) <= 1: return 0

    # Normalisation Min-Max pour que chaque objectif pèse autant
    f_min = front.min(axis=0)
    f_max = front.max(axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1  # Éviter division par zéro

    normalized_pf = (front - f_min) / denom

    d = []
    for i in range(len(normalized_pf)):
        # Distance de Manhattan sur les valeurs normalisées
        distances = np.sum(np.abs(normalized_pf - normalized_pf[i]), axis=1)
        # On prend la distance au plus proche voisin (le 0 est le point lui-même)
        d.append(np.partition(distances, 1)[1])

    d_mean = np.mean(d)
    spacing = np.sqrt(np.sum((d - d_mean) ** 2) / (len(front) - 1))
    return spacing




