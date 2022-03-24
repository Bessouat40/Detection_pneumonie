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

On voit qu'on a une precision élevée. Ce qui signifie qu'on prédit bien si une personne est malade. Par contre on détecte très mal si une personne est saine. en effet, la précision associée est de 0.26, ce qui est très faible.

D'après le recall, on voit aussi que le modèle ne prédit correctement que 63% des personnes malades contre 39% des gens pas malades. C'est encore une fois un résultat qui n'est pas satisfaisant.

Le F1-score qui est une sorte de "moyenne" entre recall et precision nous confirme que notre modèle n'est pas satisfaisant tel quel.

# Passage en niveau de gris car aucune différence avec l'image en rgb
