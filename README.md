# HessGPT - Modèle de Langage Custom

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)


**HessGPT** est un modèle de langage de type GPT entraîné from scratch, conçu pour la conversation et le suivi d'instructions.ce modèle a été entraîné sur un dataset mixte de conversations synthétiques et d'instructions (Alpaca, Dolly, WizardLM).

---

## Statistiques du Modèle

### Architecture
```
• Type: GPT (Transformer Decoder-only)
• Paramètres: ~51M
• Vocabulaire: 50,257 tokens (GPT-2)
• Dimensions d'embedding: 512
• Nombre de couches: 8
• Têtes d'attention: 8
• Contexte maximum: 1024 tokens
• Dropout: 0.05
```

### Entraînement (Epoch 6 - Meilleur Checkpoint)
```
• Validation Loss: 2.13
• Samples entraînés: 947,682
• Dataset: Synthétique 10K + Alpaca + Dolly + WizardLM
• Device: CPU/CUDA compatible
```
  
---

##  Performance Benchmark (Score sur 20)

### Note Globale: **13.3/20** - Mention: **BIEN**

| Critère | Score | Pourcentage | Statut |
|---------|-------|-------------|--------|
| **Précision Factuelle** | 5.2/8 | 65% |  Bon |
| **Qualité Conversationnelle** | 2.8/5 | 56% |  Moyen |
| **Suivi d'Instructions** | 2.5/4 | 63% |  Correct |
| **Créativité** | 1.8/2 | 90% |  Excellent |
| **Robustesse (Anti-bug)** | 1.0/1 | 100% |  Parfait |

### Résultats Détaillés par Catégorie

####  Conversations (61.7/100)
Tests sur 12 interactions sociales (salutations, remerciements, au revoir)
- **Forces**: Répond correctement à "How are you?" et aux remerciements
- **Faiblesses**: Parfois trop bref ou pas assez naturel
- **Exemple**: 
  ```
  User: "How are you?"
  HessGPT: "I'm doing great! How about you?" 
  ```

####  Questions Factuelles (80.8/100) 
Tests sur 8 questions (géographie, histoire, sciences)
- **Forces**: Excellente précision sur les capitales, dates historiques
- **Réussites notables**:
  - Capital de France:  "Paris"
  - Auteur de Roméo & Juliette:  "William Shakespeare"
  - Fin WWII:  "1945"
- **Faiblesses**: Quelques verbosités sur des réponses complexes
- **Exemple**:
  ```
  User: "What is the capital of France?"
  HessGPT: "Paris, France is the capital of France." 
  ```

####  Instructions Simples (62.0/100)
Tests sur 5 tâches (listes, énumérations)
- **Forces**: Suit bien les formats demandés (listes numérotées)
- **Faiblesses**: Parfois trop concis ou générique
- **Exemple**:
  ```
  User: "List 5 benefits of exercise."
  HessGPT: "1. Increased cardiovascular health..." 
  ```

####  Explications (95.6/100) 
Tests sur 5 concepts complexes (photosynthèse, gravité, démocratie)
- **Forces**: Capacité exceptionnelle à expliquer des concepts abstraits
- **Réussites notables**:
  - Photosynthèse:  95/100
  - Gravité:  96/100
  - Démocratie:  97/100
  - Intelligence Artificielle:  97/100
- **Point fort majeur**: Clarté pédagogique
- **Exemple**:
  ```
  User: "What is democracy?"
  HessGPT: "Democracy is a fundamental principle of voting 
  that represents and protects the interests of all citizens..." 
  ```

####  Créativité (97.0/100)
Tests sur 3 tâches créatives (haïku, descriptions, métaphores)
- **Forces**: Imagination et style poétique
- **Exemples**:
  ```
  User: "Describe a sunset in one sentence."
  HessGPT: "The sun was shining brightly, casting a warm glow 
  on the sky as it illuminated its dark blue eyes..."  99/100
  ```

#### 📝 Instructions avec Input (65.5/100)
Tests sur 2 tâches (résumé, traduction)
- **Forces**: Comprend le contexte fourni
- **Faiblesses**: Parfois dérive du sujet initial

---

##  Points Forts

 **Excellent en créativité** (97/100) - Idéal pour génération de contenu littéraire  
 **Très bon en explications** (95.6/100) - Peut servir d'assistant pédagogique  
 **Stable et robuste** - Aucun bug détecté, pas de mode collapse  
 **Bon en factuel** (80.8/100) - Fiable pour questions générales  
 **Architecture légère** - 51M paramètres, déployable sur CPU  

---

## Limitations Connues

 **Conversations basiques perfectibles** (61.7/100) - Manque parfois de naturel  
 **Verbosité occasionnelle** - Peut donner des réponses trop longues  
 **Pas de mémoire conversationnelle** - Chaque prompt est indépendant  
 **Contexte limité** - Maximum 1024 tokens  
 **Dataset anglais uniquement**    
 **pas d'historique** - pour l'instant car model trop petit

---

## Installation & Utilisation

### Prérequis
```bash
Python 3.8+
PyTorch 2.0+
transformers
flask
flask-cors
```

### Installation
```bash
git clone https://github.comtherealskyline/IAv2
cd HessGPT
pip install -r requi.txt
```

### Démarrage du Serveur Web
```bash
python app.py
```

Accédez à l'interface : `http://localhost:5000`



##  Cas d'Usage Recommandés

###  Idéal pour:
- **Génération de contenu créatif** (poèmes, descriptions, métaphores)
- **Assistant pédagogique** (explications de concepts)
- **Chatbot de support** (questions factuelles simples)
- **Prototypage rapide** d'applications IA
- **Recherche académique** sur les LLMs

###  À éviter:
- Applications critiques nécessitant 100% de précision
- Traduction professionnelle
- Analyses financières ou médicales
- Conversations longues avec contexte complexe

---

##  Évolution Future

### Mn model a + 100M parametres
Une version alternative existe avec:
-  Meilleure conversation 
-  Meilleure créativité 
- pour un pre train de 3B a 2B token filtrer et recuperer du dataset open source FineWeb
- un sft avec 1M ou 2M de dialogue pour une meilleur experience
- l'ajout des requette http avec le tool calling


##  Contribution

Les contributions sont bienvenues ! Domaines prioritaires:
- Amélioration du dataset d'entraînement pour des donner synthetique
- Optimisation de l'interface
- Tests supplémentaires
- Documentation

---

##  Remerciements

- **Datasets**: Alpaca, Dolly, WizardLM,
- **Architecture**: Inspirée de GPT-2/GPT-3.5
- **Framework**: PyTorch & Hugging Face Transformers

---

##  Contact

Pour questions ou collaborations: [silyan.silyancma@gmail.com] ou [skylineskyline59100@gmail.com]


---

## IA realiser from scratch (de zero) par:
- Silyan Larak 
- 15 ans 
- second 5 au lycee baudlaire
