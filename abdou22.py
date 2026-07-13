"""import cv2
import os

# ==============================
# PARAMÈTRES
# ==============================
TRAIN_FOLDER = "Matricules"
DATASET_FOLDER = "License Plates"
SEUIL = 25

# ==============================
# INITIALISATION
# ==============================
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

print("Chargement du dataset...")

# ==============================
# BOUCLE SUR CHAQUE IMAGE TRAIN
# ==============================
for train_file in os.listdir(TRAIN_FOLDER):

    if not train_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print("\n==============================")
    print("Test du matricule :", train_file)

    train_path = os.path.join(TRAIN_FOLDER, train_file)
    img_query = cv2.imread(train_path, 0)

    if img_query is None:
        continue

    kp_q, des_q = sift.detectAndCompute(img_query, None)

    if des_q is None:
        print("Aucun descripteur détecté.")
        continue

    best_match_image = None
    max_good_matches = 0
    best_matches = None
    best_kp_train = None
    best_img_train = None

    # ==============================
    # COMPARAISON AVEC DATASET
    # ==============================
    for file in os.listdir(DATASET_FOLDER):

        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        dataset_path = os.path.join(DATASET_FOLDER, file)
        img_train = cv2.imread(dataset_path, 0)

        if img_train is None:
            continue

        kp_t, des_t = sift.detectAndCompute(img_train, None)

        if des_t is None:
            continue

        matches = bf.knnMatch(des_q, des_t, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        print(f"Matches avec {file}: {len(good_matches)}")

        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match_image = file
            best_matches = good_matches
            best_kp_train = kp_t
            best_img_train = img_train

    # ==============================
    # DÉCISION
    # ==============================
    if max_good_matches > SEUIL:
        print("Véhicule AUTORISÉ")
        print(f"Correspondance trouvée avec : {best_match_image}")
        
        # ==============================
        # AFFICHAGE SIFT
        # ==============================
        img_matches = cv2.drawMatches(
            img_query,
            kp_q,
            best_img_train,
            best_kp_train,
            best_matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow("Matching SIFT", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Véhicule NON autorisé") """


"""import cv2
import os

# ==============================
# PARAMÈTRES
# ==============================
TRAIN_FOLDER = "Matricules"
DATASET_FOLDER = "License Plates"
SEUIL = 25
MAX_MATCH_DISPLAY = 40

# ==============================
# INITIALISATION
# ==============================
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

print("Chargement du dataset...")

# ==============================
# BOUCLE SUR CHAQUE IMAGE TRAIN
# ==============================
for train_file in os.listdir(TRAIN_FOLDER):

    if not train_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print("\n==============================")
    print("Test du matricule :", train_file)

    train_path = os.path.join(TRAIN_FOLDER, train_file)
    img_query = cv2.imread(train_path)
    gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)

    kp_q, des_q = sift.detectAndCompute(gray_query, None)

    if des_q is None:
        print("Aucun descripteur détecté.")
        continue

    best_match_image = None
    max_good_matches = 0
    best_matches = None
    best_kp_train = None
    best_img_train = None

    # ==============================
    # COMPARAISON AVEC DATASET
    # ==============================
    for file in os.listdir(DATASET_FOLDER):

        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        dataset_path = os.path.join(DATASET_FOLDER, file)
        img_train_color = cv2.imread(dataset_path)
        gray_train = cv2.cvtColor(img_train_color, cv2.COLOR_BGR2GRAY)

        kp_t, des_t = sift.detectAndCompute(gray_train, None)

        if des_t is None:
            continue

        matches = bf.knnMatch(des_q, des_t, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        print(f"Matches avec {file}: {len(good_matches)}")

        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match_image = file
            best_matches = good_matches
            best_kp_train = kp_t
            best_img_train = img_train_color

    # ==============================
    # DÉCISION
    # ==============================
    if max_good_matches > SEUIL:

        print("Véhicule AUTORISÉ")
        print(f"Correspondance trouvée avec : {best_match_image}")

        # Dessiner les correspondances
        img_matches = cv2.drawMatches(
            img_query,
            kp_q,
            best_img_train,
            best_kp_train,
            best_matches[:MAX_MATCH_DISPLAY],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Ajouter texte sur l'image
        text = f"Match: {best_match_image} | Points: {max_good_matches}"
        cv2.putText(img_matches, text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Redimensionner si image trop grande
        height, width = img_matches.shape[:2]
        if width > 1400:
            scale = 1400 / width
            img_matches = cv2.resize(img_matches, None, fx=scale, fy=scale)

        cv2.imshow("SIFT Matching Result", img_matches)
        cv2.waitKey(800)   # ferme automatiquement après 0.8 sec

    else:
        print("Véhicule NON autorisé")

cv2.destroyAllWindows() """



import cv2
import os

# ==============================
# PARAMÈTRES
# ==============================
TRAIN_FOLDER = "Matricules"
TEST_FOLDER = "License Plates"
THRESHOLD = 25

# ==============================
# INITIALISATION SIFT
# ==============================
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()   # sans crossCheck

# ==============================
# LECTURE DES IMAGES TRAIN
# ==============================
train_images = {}
train_descriptors = {}
train_keypoints = {}

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
        train_keypoints[file] = kp

print("Dataset TRAIN chargé.")

# ==============================
# TRAITEMENT DES IMAGES TEST
# ==============================
for file in os.listdir(TEST_FOLDER):

    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

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
    best_good_matches = None

    # ==============================
    # COMPARAISON AVEC TRAIN
    # ==============================
    for train_name, des_train in train_descriptors.items():

        if des_train is None:
            continue

        matches = bf.knnMatch(des_test, des_train, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print(f"Matches avec {train_name}: {len(good)}")

        if len(good) > best_match_count:
            best_match_count = len(good)
            best_image_name = train_name
            best_good_matches = good

    # ==============================
    # DÉCISION + AFFICHAGE
    # ==============================
    if best_match_count > THRESHOLD:

        print("Véhicule AUTORISÉ")
        print("Correspondance trouvée avec :", best_image_name)

        img_matches = cv2.drawMatches(
            test_img,
            kp_test,
            train_images[best_image_name],
            train_keypoints[best_image_name],
            best_good_matches[:30],   # afficher les 30 meilleurs
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow("Matching SIFT", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Véhicule NON autorisé")