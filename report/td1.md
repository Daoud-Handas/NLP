# Rapport TD1

# Partie 1 

## Features 
Liste des features utilisés : 
- **stopwords** : liste de stopwords
- **stemming** : racinisation des mots
- **tokenization** : segmentation des mots
- **lowercase** : mise en minuscule des mots
- **n-grams** : bigrams (ex : "Je mange", "mange une", "une pomme"...)
- **n-gams range** : intervalle de n-grams (ex : 1-grams, 2-grams, 3-grams...)

## Models
- Logistic Regression
- Random Forest
- Linear Model

## Resultats
- Résulat Linear Model : nan% -> Les métriques de classification ne peuvent pas gérer un mélange de cibles binaires et continues.
- Résultat Logistic Regression : ~91%
- Résultat Random Forest : ~90.5%
- Résultat Logistic Regression
  - stopwords : ~91.1%
  - stemming : ~91.1%
  - tokenization : ~91.7%
  - lowercase : ~91.7%
  - n-grams (trigrams) : ~93%
  - n-grams range(1-4) : ~93.1%
- Résulats Random Forest
  - stopwords : ~91.2%
  - stemming : ~90.5%
  - tokenization : ~90.5%
  - lowercase : ~90.8%
  - n-grams (trigrams) : ~91.5%
  - n-grams range(1-4) : ~91.7%

## Erreurs rencontrées
Attribute error : 'str' object has no attribute 'lower' :
- Erreur : on essaye d'appliquer la méthode lower() à un string
- Solution : pour le cas du n-grams, retourner le résultat sous cette forme : **return ' '.join([' '.join(grams) for grams in n_grams])**

## Conclusion

### A-priori features/modèles 
- Les meilleurs résultats sont obtenus avec les n-grams range(1-4) et la régression logistique. Nous pension que le Random Forest serait plus performant que la régression logistique mais ce n'est pas le cas.
- Le lowercase et le stemming ne semblent pas avoir d'impact sur les résultats.
- Le linear model ne semble pas fonctionner avec les données de ce dataset.

### Apports individuels
- Les n-grams range(1-4) sont les plus performants.
- La régression logistique est le modèle le plus performant.

### Conclusion sur le bon modeling
**Le modèle le plus performant est la régression logistique avec les n-grams range(1-4).**


# Partie 2


Ce rapport traite de l'extraction de caractéristiques, de l'expérimentation de différents modèles, et des conclusions sur la modélisation actuelle.

## A-priori sur les Caractéristiques et les Modèles

Au départ, nous avions des a-priori sur les caractéristiques et les modèles qui pourraient être utiles pour la tâche spécifique que nous cherchions à résoudre. Voici quelques-unes de nos hypothèses initiales :

### Caractéristiques
1. La conversion en minuscules (lowercasing) pourrait être utile pour normaliser le texte.
2. La suppression des stopwords pourrait réduire le bruit dans les données.
3. La racinisation (stemming) pourrait aider à regrouper des mots apparentés.
4. La tokenisation pourrait diviser le texte en mots individuels pour une analyse plus fine.
5. L'utilisation de caractéristiques telles que "is_starting_word", "is_final_word", "is_capitalized" et "is_punctuation" pourrait améliorer la performance du modèle.

### Modèles
1. Les modèles tels que la régression logistique, les forêts aléatoires et les modèles linéaires pourraient être des choix appropriés pour cette tâche.
2. L'utilisation du POS tagging (Part-of-Speech tagging) en français pourrait apporter des informations grammaticales précieuses pour la classification des mots.

## Apports des Variations

Nous avons expérimenté différentes variations des caractéristiques et des modèles pour évaluer leur impact sur la performance du modèle. Voici les apports individuels de certaines de ces variations :

### Caractéristiques
- **Conversion en Minuscules (Lowercasing)** : Cette variation a contribué à la normalisation du texte, améliorant la performance en général en évitant les doublons de mots dus à la casse.
- **Suppression des Stopwords** : Cette variation a nettoyé le texte de mots courants inutiles, améliorant la précision de la classification.
- **Racinisation (Stemming)** : Cette variation a permis de regrouper les mots apparentés, réduisant la dimensionnalité des données et améliorant la généralisation.
- **Tokenisation** : Cette variation a permis d'analyser chaque mot individuellement, permettant au modèle de prendre en compte les caractéristiques spécifiques des mots.
- **Caractéristiques Additionnelles** : L'ajout de caractéristiques telles que "is_starting_word", "is_final_word", "is_capitalized" et "is_punctuation" a permis au modèle de prendre en compte des informations plus fines pour la classification.

### Modèles
- **Régression Logistique** : La régression logistique a fourni un modèle simple et interprétable, mais n'a pas bien géré la complexité de la tâche.
- **Random Forest** : Les forêts aléatoires ont montré une meilleure capacité de généralisation et ont bien performé en utilisant les caractéristiques améliorées.
- **Modèles Linéaires** : Les modèles linéaires n'ont pas bien performé en raison de la nature complexe de la tâche.
- **POS Tagging** : L'utilisation du POS tagging s'est avérée difficile en français en raison du manque de ressources, et n'a pas apporté d'amélioration significative.

## Conclusion sur le Bon Modélisation (en l'État)

Sur la base de nos expérimentations, la modélisation actuelle recommandée pour cette tâche serait l'utilisation de forêts aléatoires avec des caractéristiques telles que la conversion en minuscules, la suppression des stopwords, la racinisation, la tokenisation, et les caractéristiques additionnelles comme "is_starting_word", "is_final_word", "is_capitalized", et "is_punctuation". Cette combinaison de caractéristiques et de modèle a démontré une bonne capacité de généralisation et de classification précise pour la tâche en cours.

Cependant, il convient de noter que l'amélioration continue est possible en explorant d'autres modèles et caractéristiques, ainsi qu'en évaluant l'ajout de ressources linguistiques pour le français. La modélisation est un processus itératif, et des ajustements ultérieurs peuvent être nécessaires pour améliorer davantage les performances du modèle.