
import matplotlib.pyplot as plt
import numpy as np
import json
import pygmo as pg


def enregistrer_json(moead, filename="resultats_pareto"):
    data_to_save = []
    filename =filename+".json"
    # On déballe le tuple (chromosome, scores) directement ici ! Grrr.
    for chromo, scores in moead.ex_pop:
        data_to_save.append({
            "chromosome": chromo.tolist() if isinstance(chromo, np.ndarray) else chromo,
            "objectives": {
                "F1_temps": scores[0],
                "F2_energie": scores[1],
                "F3_charge": scores[2]
            }
        })

    with open("results/"+filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f"Archive sauvegardée dans results/{filename} !")




def afficher_pareto_3d(moead, name_file):
    scores = []

    name_file = name_file+".png"
    for chromo, f_scores in moead.ex_pop:
        # Pas besoin d'appeler mop.evaluate ! On les a déjà. Grrr.
        scores.append(f_scores)



    scores = np.array(scores)

    if len(scores) == 0:
        print("VIDE")
        return

    # Extraction des 3 objectifs
    f1_temps = scores[:, 0]
    f2_energie = scores[:, 1]
    f3_charge = scores[:, 2]

    # Création de la figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Dessin des points (Le Front de Pareto)
    p = ax.scatter(f1_temps, f2_energie, f3_charge, c=f2_energie, cmap='viridis', s=50)

    # Titres et labels issus de ton rapport [cite: 116, 120]
    ax.set_title("Front de Pareto : Allocation MEC/Cloud")
    ax.set_xlabel("F1 : Temps Total (ms)")
    ax.set_ylabel("F2 : Énergie (W)")
    ax.set_zlabel("F3 : Équilibrage (MSU)")

    fig.colorbar(p, ax=ax, label='Intensité Énergétique')
    print(f"save in results/{name_file}")
    plt.savefig("results/" + name_file)  # Sauvegarde l'image
    try :
        plt.show()

    finally :

        plt.close(fig)  # Libère la mémoire


def afficher_pareto(pop, name_file):
    # PyGMO nous donne directement tous les scores (objectifs)
    # fits est une matrice de forme (taille_pop, 3) grrr
    scores = pop.get_f()

    if len(scores) == 0:
        print("VIDE... grrr")
        return

    # Si tu ne veux afficher QUE les solutions non-dominées (le vrai front)
    # ndf contient les indices des fronts. ndf[0] est le meilleur front.
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(scores)
    pareto_scores = scores[ndf[0]]

    # Extraction des 3 objectifs grrr
    f1_temps = pareto_scores[:, 0]
    f2_energie = pareto_scores[:, 1]
    f3_charge = pareto_scores[:, 2]

    # Création de la figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Dessin des points grrr
    p = ax.scatter(f1_temps, f2_energie, f3_charge, c=f2_energie, cmap='viridis', s=50)

    # Titres et labels
    ax.set_title("Front de Pareto : Allocation MEC/Cloud (MOEA/D)")
    ax.set_xlabel("F1 : Temps Total (ms)")
    ax.set_ylabel("F2 : Énergie (W)")
    ax.set_zlabel("F3 : Équilibrage (MSU)")

    fig.colorbar(p, ax=ax, label='Intensité Énergétique')

    if not name_file.endswith(".png"):
        name_file += ".png"

    print(f"Sauvegarde dans results/{name_file}... grrr")
    plt.savefig("results/" + name_file)

    try:
        plt.show()
    finally:
        plt.close(fig)




import pandas as pd
import plotly.express as px

def plot_parallel_coordinates(pf,filename):
    filename = "parallel_"+filename+".png"
    df = pd.DataFrame(pf, columns=['Temps (F1)', 'Énergie (F2)', 'MSU (F3)'])
    fig = px.parallel_coordinates(df, color='Temps (F1)',
                             labels={"Temps (F1)": "Time",
                                     "Énergie (F2)": "Energy", "MSU (F3)": "Load Variance"},
                             color_continuous_scale=px.colors.diverging.Tealrose)

    fig.write_image("results/" + filename)



