# Rapport TD1

## Features 
Liste des features utilisés : 
- **stopwords** : liste de stopwords
- **stemming** : racinisation des mots
- **tokenization** : segmentation des mots
- **lowercase** : mise en minuscule des mots
- **n-grams** : bigrams (ex : "Je mange", "mange une", "une pomme"...)
- **range** : intervalle de n-grams (ex : 1-grams, 2-grams, 3-grams...)

## Models

## Results

## Erreurs rencontrées
Attribute error : 'str' object has no attribute 'lower' :
- Erreur : on essaye d'appliquer la méthode lower() à un string
- Solution : pour le cas du n-grams, retourner le résultat sous cette forme : **return ' '.join([' '.join(grams) for grams in n_grams])**
