# Moteur de Recherche

## Binome: Chaima Aouichi - Mohamed Aziz Cherif

Un moteur de recherche d'image qui se base sur les descripteurs de texture et les histogrammes de couleurs

### Explication du contenu des fichiers

-utils.py: Contient les fonctions qui permettent de génerer les descripteurs de texture et de couleur des differentes images et de mesurer les similarités entre eux

-img_descriptors.json: Contient les descripteurs de couleur de toutes les images

-img_indices.json: Contient les descripteurs de texture de toutes les images

-app.py: Le code de l'application finale (On a utilisé PyQt5 pour génerer l'interface graphique)

### Utilisation

1ère étape: Il faut installer les librairies nécessaires en éxecutant: pip install -r requirement.txt

2ème étape: Dans app.py: il faut changer le chemin (path) dans la variable `base_path` par le chemin où vous avez cloné le repertoire du projet

3ème étape: Lancer app.py
