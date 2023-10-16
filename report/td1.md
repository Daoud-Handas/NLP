# Rapport TD1

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