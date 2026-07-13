"""
import cv2
import os
import numpy as np

# ===============================
# 1. Initialisation SIFT
# ===============================

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)

# ===============================
# 2. Charger dataset TRAIN
# ===============================

train_folder = "matricules"
train_images = os.listdir(train_folder)

train_descriptors = {}
train_keypoints = {}

for img_name in train_images:
    img_path = os.path.join(train_folder, img_name)
    img = cv2.imread(img_path, 0)
    
    kp, des = sift.detectAndCompute(img, None)
    
    train_descriptors[img_name] = des
    train_keypoints[img_name] = kp

print("Dataset TRAIN chargé.")

# ===============================
# 3. Tester chaque voiture (QUERY)
# ===============================

query_folder = "License Plates"
query_images = os.listdir(query_folder)

for query_name in query_images:
    
    print("\n==============================")
    print("Test de :", query_name)
    
    query_path = os.path.join(query_folder, query_name)
    query_img = cv2.imread(query_path, 0)
    
    kp_query, des_query = sift.detectAndCompute(query_img, None)
    
    best_match_count = 0
    best_match_name = None
    
    # ===============================
    # 4. Comparer avec chaque plaque TRAIN
    # ===============================
    
    for train_name in train_images:
        
        des_train = train_descriptors[train_name]
        
        if des_train is None or des_query is None:
            continue
        
        matches = bf.knnMatch(des_query, des_train, k=2)
        
        good = []
        
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        print(f"Matches avec {train_name}: {len(good)}")
        
        if len(good) > best_match_count:
            best_match_count = len(good)
            best_match_name = train_name
    
    # ===============================
    # 5. Décision finale
    # ===============================
    
    SEUIL = 25
    
    if best_match_count > SEUIL:
        print("Véhicule AUTORISÉ")
        print("Correspondance trouvée avec :", best_match_name)
    else:
        print("Véhicule NON autorisé") 
        
"""


import cv2
import os
import numpy as np

# ==============================
# PARAMÈTRES
# ==============================
TRAIN_FOLDER = "Matricules"
TEST_FOLDER = "License Plates"
THRESHOLD = 25   # seuil de décision (à ajuster)

# ==============================
# INITIALISATION SIFT
# ==============================
sift = cv2.SIFT_create()

# ==============================
# LECTURE DES IMAGES TRAIN
# ==============================
train_images = {}
train_descriptors = {}

print("Chargement des images TRAIN...")

for file in os.listdir(TRAIN_FOLDER):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(TRAIN_FOLDER, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        kp, des = sift.detectAndCompute(img, None)

        train_images[file] = img
        train_descriptors[file] = des

print("Dataset TRAIN chargé.")

# ==============================
# TRAITEMENT DES IMAGES TEST
# ==============================
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

for file in os.listdir(TEST_FOLDER):

    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue   # Ignore Thumbs.db

    print("\n==============================")
    print("Test de :", file)

    test_path = os.path.join(TEST_FOLDER, file)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if test_img is None:
        continue

    kp_test, des_test = sift.detectAndCompute(test_img, None)

    if des_test is None:
        print("Aucun descripteur détecté.")
        continue

    best_match_count = 0
    best_image_name = None
    best_matches = None

    # Comparaison avec toutes les images TRAIN
    for train_name, des_train in train_descriptors.items():

        if des_train is None:
            continue

        matches = bf.match(des_test, des_train)
        matches = sorted(matches, key=lambda x: x.distance)

        match_count = len(matches)

        print(f"Matches avec {train_name}: {match_count}")

        if match_count > best_match_count:
            best_match_count = match_count
            best_image_name = train_name
            best_matches = matches

    # ==============================
    # DÉCISION
    # ==============================
    if best_match_count > THRESHOLD:
        print("Véhicule AUTORISÉ")
        print("Correspondance trouvée avec :", best_image_name)

        # Affichage des 30 meilleurs matches
        img_matches = cv2.drawMatches(
            test_img,
            kp_test,
            train_images[best_image_name],
            sift.detectAndCompute(train_images[best_image_name], None)[0],
            best_matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow("Matching SIFT", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Véhicule NON autorisé")