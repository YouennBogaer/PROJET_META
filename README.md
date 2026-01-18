# Optimisation Multi-objectif : Allocation Cloud & MEC
### ANDRE V./BOGAER Y./DANO E. (Also know as "les tigres")

Ce projet implémente MOEA/D pour de l'allocation de ressources

---

## Installation de l'environnement

Le projet utilise Python 3.9. Il est impératif d'utiliser Conda pour garantir la compatibilité des modules.

### Configuration de l'environnement

```bash
# 1. Création de l'environnement
conda create -n env python=3.9 -y

# 2. Activation de l'environnement
conda activate env

# 3. Installation des dépendances
pip install -r requirements.txt
```
ou 
```bash
# Création de l'environnement à partir du fichier YAML
conda env create -f environment.yml

# Activation de l'environnement (le nom est défini à l'intérieur du fichier .yml)
conda activate opti
```
---
## Guide d'exécution

Le projet est structuré autour de trois scripts principaux selon l'analyse souhaitée.

### 1. Exécution unitaire (main.py)
Lance une seule exécution de l'algorithme MOEA/D. Ce script est idéal pour visualiser le front de Pareto et les métriques (Spacing, Hypervolume) d'un test spécifique.

```bash
python main.py --name mon_resultat --stop 500 --save_json --H 15 --mutation 0.5
```

**Arguments disponibles :**
- --name : Nom de base pour les fichiers de sortie (resultat par défaut).
- --stop : Nombre de générations / critère d'arrêt (1500 par défaut).
- --save_json : Enregistre les solutions et scores dans un fichier JSON.
- --mutation : Taux de mutation par enfant (0.5 par défaut).
- --H : Nombre de découpages des sous-problèmes (15 par défaut).
- --neighbor : Ratio de voisinage (0.05 par défaut).

### 2. Grid Search (main_cross_validation.py)
Exécute une recherche par grille sur plusieurs combinaisons d'hyperparamètres. Ce script utilise le multiprocessing pour optimiser le temps de calcul.

```bash
python main_cross_validation.py
```
- **Sortie** : Génère un fichier grid_search.csv récapitulant les performances de chaque configuration dans le dossier results_parameters/.

### 3. Comparaison NSGA-II (NGSA_II_main.py)
Script dédié à l'exécution de l'algorithme NSGA-II. Il permet de collecter des données de performance pour comparer cette approche avec MOEA/D.

```bash
python NGSA_II_main.py
```

---

## Structure des résultats

Les fichiers de sortie sont organisés comme suit :

- **/results** : 
    - Graphiques 3D du front de Pareto (Objectifs : Temps, Énergie, Charge).
    - Visualisations en coordonnées parallèles.
    - Courbes d'évolution EP (Pareto Front size) et HP (Hypervolume).
- **/results_parameters** : 
    - Fichiers CSV issus des recherches de paramètres (Grid Search).

- **/results_comparaisons_plot** : 
    - Images pour comparer les 3 fonctions objectives avec différnets nombres de tâches


## NB
Le code a été éxécuté surr unbuntu pour gérer le multi-processing mais on ne sait pas si cela marche sur Windows ou MacOs.


---


