# Évaluation des modèles de vision par ordinateur

Homework **ML Evaluation – Master Data & IA (INPT, 2026)** : métriques d'évaluation de trois tâches de vision.

| Tâche | Métriques principales |
|---|---|
| Détection d'objets | Précision, rappel, F1, AP@0.5, AP@0.75, **mAP@[.5:.95]** (norme COCO) |
| Segmentation sémantique | **mIoU**, Dice, précision pixel |
| Segmentation d'instance | **AP masque**, et PQ (Panoptic Quality) en complément |

Le fil conducteur est l'**IoU** (Intersection over Union), commune à toutes ces métriques.

## Contenu du dépôt

```
.
├── notebook_evaluation_vision.ipynb   # tout le code
├── requirements.txt
├── Evaluation_Metrics_Vision.pptx     # présentation
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
jupyter notebook notebook_evaluation_vision.ipynb
```

Sur Google Colab, activer un GPU (Exécution → Modifier le type d'exécution). Exécuter les cellules **dans l'ordre**.
La partie 2 (métriques) fonctionne sans GPU et sans PyTorch.

## Documentation du code

### Partie 0 : configuration
Imports, détection automatique de PyTorch et du périphérique (`cuda`/`cpu`), `N_IMAGES = 40` (nombre d'images évaluées en partie 3, 170 au maximum), fonction `download(url, dest)` et utilitaires de couleurs.

### Partie 1 : voir ce que produit chaque tâche
Sur l'image d'exemple `bus.jpg`, deux modèles pré-entraînés sont appliqués :
- **Mask R-CNN** (`maskrcnn_resnet50_fpn_v2`) : détection (boîtes, classe, score) et segmentation d'instance (un masque par objet) ;
- **DeepLabV3** (`deeplabv3_resnet50`) : segmentation sémantique (une classe par pixel).

La figure est enregistrée dans `figure_taches.png`.

### Partie 2 : métriques codées à la main (NumPy)

| Fonction | Rôle |
|---|---|
| `prf(tp, fp, fn)` | Précision, rappel, F1 |
| `box_iou(a, b)` | Matrice d'IoU entre deux ensembles de boîtes `(x1, y1, x2, y2)` |
| `mask_iou(a, b)` | Matrice d'IoU entre deux ensembles de masques booléens `(N, H, W)` |
| `match_detections(scores, ious, thr)` | Appariement glouton (règle COCO) : détections triées par score, chacune prend le meilleur objet réel libre si IoU ≥ seuil. Renvoie `(est_TP, indice_apparié)` |
| `average_precision(preds, gts, iou_fn, thr)` | AP à un seuil d'IoU : courbe précision-rappel, enveloppe décroissante, 101 niveaux de rappel. Renvoie `(AP, rappel, précision)` |
| `coco_ap(preds, gts, iou_fn)` | Moyenne sur les 10 seuils d'IoU 0,50 à 0,95 : renvoie `mAP`, `AP50`, `AP75` |
| `confusion_matrix(gt, pred, n_classes)` | Matrice de confusion (lignes = vérité terrain) |
| `seg_metrics(cm)` | Précision pixel, mIoU, IoU et Dice de l'objet |
| `pq_counts(...)`, `pq_scores(...)` | Panoptic Quality : PQ = SQ × RQ |

`average_precision` et `coco_ap` fonctionnent aussi bien pour la détection (`iou_fn=box_iou`) que pour l'instance (`iou_fn=mask_iou`).

Exemples vérifiables à la main :
- IoU de deux boîtes 10×10 décalées de (5, 5) : 25 / 175 ≈ 0,14 ;
- exemple de détection : 3 images, 4 objets réels, 6 détections ;
- précision pixel trompeuse : un modèle qui répond « fond » partout a 99 % de précision pixel mais un mIoU d'environ 50 % ;
- PQ : 1 TP, 1 FP, 2 FN, d'où RQ = 1 / 2,5 = 0,4.

### Partie 3 : évaluation réelle
Jeu de données **Penn-Fudan** (170 images de piétons, avec boîtes, masques d'instance et masques sémantiques), téléchargé automatiquement dans `data/`. Les modèles, pré-entraînés sur COCO, sont évalués **sans entraînement** sur la classe « personne » :
- Mask R-CNN (boîtes + masques), SSDlite320 MobileNetV3 (boîtes, léger), DeepLabV3 (sémantique).

Sorties : tableau de métriques (mAP, AP@0.5, AP@0.75, AP masque, PQ, mIoU, Dice), courbes précision-rappel, graphique précision / vitesse (latence, FPS, nombre de paramètres).

## Limites
- Avec 40 images et une seule classe, les chiffres **illustrent la méthode** : ce n'est pas un benchmark.
- Les piétons non annotés dans Penn-Fudan comptent comme faux positifs.
- Les vitesses dépendent du matériel utilisé.

## Références
Lin et al., *Microsoft COCO* (2014) · Everingham et al., *Pascal VOC* (2010) · Kirillov et al., *Panoptic Segmentation* (2019) · He et al., *Mask R-CNN* (2017) · Chen et al., *DeepLab* (2018).
