import cv2
import os

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

img_query = cv2.imread('matricule1.png', 0)
kp_q, des_q = sift.detectAndCompute(img_query, None)

plates = "License Plates"
best_match_image = None
max_good_matches = 0

for file in os.listdir(plates):
    img_train = cv2.imread(os.path.join(plates, file), 0)
    if img_train is None: continue
    
    kp_t, des_t = sift.detectAndCompute(img_train, None)
    
    matches = bf.knnMatch(des_q, des_t, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) > max_good_matches:
        max_good_matches = len(good_matches)
        best_match_image = file

print(f"Le véhicule est identifié comme : {best_match_image} avec {max_good_matches} points communs.")