import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

# ===============================
# PARAMÈTRES
# ===============================
QUERY_FOLDER = "Matricules"
TRAIN_FOLDER = "License Plates"
SEUIL = 15          # seuil de décision
RATIO = 0.75        # ratio test Lowe

# ===============================
# INITIALISATION
# ===============================
sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2)

print("Chargement des images TRAIN...")

# ===============================
# CHARGEMENT DATASET TRAIN
# ===============================
train_data = []

for filename in os.listdir(TRAIN_FOLDER):

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(TRAIN_FOLDER, filename)
    img = cv.imread(path, 0)

    if img is None:
        continue

    kp, des = sift.detectAndCompute(img, None)

    if des is not None:
        train_data.append((filename, img, kp, des))

print("Nombre d'images train chargées :", len(train_data))

# ===============================
# TRAITEMENT DES QUERIES
# ===============================
for qname in os.listdir(QUERY_FOLDER):

    if not qname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print("\n==============================")
    print("Traitement de :", qname)

    qpath = os.path.join(QUERY_FOLDER, qname)
    query_img = cv.imread(qpath, 0)

    if query_img is None:
        continue

    kp1, des1 = sift.detectAndCompute(query_img, None)

    if des1 is None:
        print("Aucun descripteur trouvé")
        continue

    best_score = 0
    best_match = None

    # ===============================
    # COMPARAISON AVEC TRAIN
    # ===============================
    for (train_name, train_img, kp2, des2) in train_data:

        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < RATIO * n.distance:
                good.append(m)

        if len(good) > best_score:
            best_score = len(good)
            best_match = (train_name, train_img, kp2, good)

    # ===============================
    # DÉCISION
    # ===============================
    print("Meilleur score :", best_score)

    if best_score > SEUIL:

        print("Vehicule AUTORISE")
        print("Correspondance avec :", best_match[0])

        train_name, train_img, kp2, good_matches = best_match

        result = cv.drawMatches(query_img, kp1,
                                train_img, kp2,
                                good_matches[:30], None,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(12,6))
        plt.imshow(result, cmap="gray")
        plt.title(f"Matching SIFT - {train_name} ({best_score} matches)")
        plt.axis("off")
        plt.show()

    else:
        print("Vehicule NON autorise")

print("\nFin du programme.")