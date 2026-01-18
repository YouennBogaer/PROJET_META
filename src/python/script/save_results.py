import matplotlib.pyplot as plt
import numpy as np
import json
import pygmo as pg


def enregistrer_json(moead, filename="resultats_pareto"):
    """ Sauvegarde les solutions et scores en format JSON """
    data_to_save = []
    filename = filename + ".json"

    for chromo, scores in moead.ex_pop:
        data_to_save.append({
            "chromosome": chromo.tolist() if isinstance(chromo, np.ndarray) else chromo,
            "objectives": {
                "F1_temps": scores[0],
                "F2_energie": scores[1],
                "F3_charge": scores[2]
            }
        })

    with open("results/" + filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f"Archive sauvegardée dans results/{filename} !")


def afficher_pareto_3d(moead, name_file):
    """ Trace le front de Pareto 3D pour MOEA/D """
    scores = []
    name_file = name_file + ".png"

    for chromo, f_scores in moead.ex_pop:
        scores.append(f_scores)

    scores = np.array(scores)

    if len(scores) == 0:
        print("Erreur : liste de scores vide")
        return

    f1_temps = scores[:, 0]
    f2_energie = scores[:, 1]
    f3_charge = scores[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(f1_temps, f2_energie, f3_charge, c=f2_energie, cmap='viridis', s=50)

    ax.set_title("Front de Pareto : Allocation MEC/Cloud")
    ax.set_xlabel("F1 : Temps Total (ms)")
    ax.set_ylabel("F2 : Énergie (W)")
    ax.set_zlabel("F3 : Équilibrage (MSU)")

    fig.colorbar(p, ax=ax, label='Intensité Énergétique')
    print(f"Image enregistrée : results/{name_file}")
    plt.savefig("results/" + name_file)

    try:
        plt.show()
    finally:
        plt.close(fig)


def afficher_pareto(pop, name_file):
    """ Affiche les solutions non-dominées de la population NSGA-II """
    scores = pop.get_f()

    if len(scores) == 0:
        print("Erreur : population vide")
        return

    # Tri pour ne garder que le meilleur front
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(scores)
    pareto_scores = scores[ndf[0]]

    f1_temps = pareto_scores[:, 0]
    f2_energie = pareto_scores[:, 1]
    f3_charge = pareto_scores[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(f1_temps, f2_energie, f3_charge, c=f2_energie, cmap='viridis', s=50)

    ax.set_title("Front de Pareto : Allocation MEC/Cloud NSGA-II")
    ax.set_xlabel("F1 : Temps Total (ms)")
    ax.set_ylabel("F2 : Énergie (W)")
    ax.set_zlabel("F3 : Équilibrage (MSU)")

    fig.colorbar(p, ax=ax, label='Intensité Énergétique')

    if not name_file.endswith(".png"):
        name_file += ".png"

    print(f"Sauvegarde du graphique : results/{name_file}")
    plt.savefig("results/" + name_file)

    try:
        plt.show()
    finally:
        plt.close(fig)

