# Explication du projet

Le but est de prédire si un patient est atteint d'une pneumonie à partir d'une radiographie du thorax.
Pour cela, nous allons utiliser des méthodes de machine learning ainsi que des méthodes de deep learning.
A la fin du projet, j'ai utilisé un réseau de neurones qui permettra de faire de la sélection de paramètres qui seront ensuite envoyés dans un algorithme de machine learning.

# Utilisation de classes pour simplifier l'appel de méthodes de machine/deep learning et évaluer leurs performances à l'aide de plusieurs métriques d'évaluation

Sachant qu'on refait souvent les mêmes choses en pré-traitement des données et en entraînement de modèles de machine/deep learning, j'ai décidé de mettre sous forme de classes tout ce qui peut être automatisé. De cette manière, on peut appeler simplement les fonctions et on obtient une routine d'exécution du code. Pour finir, on peut évaluer les performances de nos modèles à l'aide de plusieurs métriques : recall, accuracy,...

# Librairies nécessaires

- matplotlib
- pandas
- os 
- numpy
- opencv
- joblib
- tensorflow
- sklearn
- keras

# Lancement du code

Mettre les fichier main*.py dans le même emplacement que le fichier model.py et exécuter en ligne de commande le code main*.py.
