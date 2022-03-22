# Detection_pneumonie
Utilisation de deep/machine learning pour prédire à partir d'une radio du thorax si un patient est atteint d'une pneumonie ou non.

# Base de données

Vous pouvez retrouver la base de données utilisée sur ce site : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Problème dans ce jeu de données 

La base de données est mal balancée, en effet il y a plus d'exemples de patients malades que de patients sains.
On ne peut donc pas utiliser l'accuracy comme indicateur de performance de notre modèle.

## Solution 

Pour résoudre ce problème, on utilise d'autres métriques d'évaluation telles que la précision, le rappel ou la F measure.

# Résultats obtenus à l'aide d'un réseau de neurones convolutif (couches de convolution, couches de maxpooling)

## Résultats de l'entrainement du modèle :

![screenshot1](https://github.com/Bessouat40/Detection_pneumonie/blob/main/cnn_train_result.png?raw=true)

![screenshot1](https://github.com/Bessouat40/Detection_pneumonie/blob/main/cnn_hist1.png?raw=true)

![screenshot1](https://github.com/Bessouat40/Detection_pneumonie/blob/main/cnn_hist2.png?raw=true)

## Evaluation du modèle entrainé :

![screenshot1](https://github.com/Bessouat40/Detection_pneumonie/blob/main/cnn_train_metrics.png?raw=true)

![screenshot1](https://github.com/Bessouat40/Detection_pneumonie/blob/main/cnn_interpretation.png?raw=true)
