# Optimisation Multi objectif
### ANDRE V. BOGAER Y. DANO E.

MOAED.py : tentative MOEAD from scratch

PAES_test.ipynb : tentative PAES from scratch

### Devrait tester nos resultats grace au framework open source suivant : https://github.com/moead-framework/framework
### Attention framework fonctionne en python 3.9

# Lancement

## Modules

Installez les modules nécessaires

```bash
pip install -r .\requirements.txt
```
## Execution

L'exécution principale se fait via le fichier `main.py`.
les résulats se trouvent dans le dossier results si y'en a pas il est créé automatiquement

```bash
python main.py [options]
```
## Args
--name : Définit le nom de base pour tes fichiers de sortie. Par défaut, il s'appelle resultat. Les fichiers générés (image et JSON) utiliseront ce nom dans le dossier /results.

--stop : Définit le nombre total de générations pour l'algorithme. C'est le critère d'arrêt qui détermine la durée de l'optimisation. Par défaut : 100.

--save_json : USi cet argument est présent, le script enregistre les solutions et les scores dans un fichier .json.

## Example
```bash
python main.py --name test --stop 200 --save_json
```
