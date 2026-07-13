import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt


sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2)

query_folder = "Matricules"
train_folder = "License Plates/"


# Charger les matricules et les voitures
train_images = []
train_kp_des = []

for filename in os.listdir(train_folder):
    path = os.path.join(train_folder, filename)
    img = cv.imread(path, 0)

    if img is None:
        continue

    kp, des = sift.detectAndCompute(img, None)

    if des is not None:
        train_images.append(filename)
        train_kp_des.append((img, kp, des))

print("Nombre d'images train chargées :", len(train_images))



# tester chaque matricule

for qname in os.listdir(query_folder):

    print("\n==============================")
    print("Traitement de :", qname)

    qpath = os.path.join(query_folder, qname)
    query_img = cv.imread(qpath, 0)

    if query_img is None:
        continue

    kp1, des1 = sift.detectAndCompute(query_img, None)

    if des1 is None:
        print("Aucun descripteur trouvé")
        continue

    # affichage des points SIFT de chaque matricule
    img_kp = cv.drawKeypoints(query_img, kp1, None,
                              flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    """plt.figure(figsize=(6,4))
    plt.imshow(img_kp, cmap="gray")
    plt.title("Points SIFT - Query")
    plt.axis("off")
    plt.show()"""

    best_match_name = None
    best_score = 0
    best_matches = None
    best_train_img = None
    best_kp2 = None

   
    # comparaison avec toutes les voitures
    
    for i, (train_img, kp2, des2) in enumerate(train_kp_des):

        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > best_score:
            best_score = len(good)
            best_match_name = train_images[i]
            best_matches = good
            best_train_img = train_img
            best_kp2 = kp2

    print("Meilleure correspondance :", best_match_name)
    print("Nombre de bons matches :", best_score)

   
    # decision
    
    SEUIL = 10

    if best_score < SEUIL:
        print("Vehicule autorise")
    else:
        print("Vehicule NON autorise")

    
    # affichage Matching
    
    if best_matches is not None:

        result = cv.drawMatches(query_img, kp1,
                                best_train_img, best_kp2,
                                best_matches, None,
                                flags=2)

        plt.figure(figsize=(12,6))
        plt.imshow(result)
        plt.title("Meilleure Correspondance SIFT")
        plt.axis("off")
        plt.show()