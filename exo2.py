import cv2
import numpy as np
from matplotlib import pyplot as plt


query_img = cv2.imread('im1.png', 0)   
train_img = cv2.imread('im1.jpg', 0)   


sift = cv2.SIFT_create()


kp1, des1 = sift.detectAndCompute(query_img, None)
kp2, des2 = sift.detectAndCompute(train_img, None)

print("Nombre de keypoints (query):", len(kp1))
print("Nombre de keypoints (train):", len(kp2))


bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Nombre de bons matches:", len(good_matches))


SEUIL = 20

if len(good_matches) > SEUIL:
    print("Objet trouvé dans l'image modèle")
else:
    print("Objet NON trouvé")


result = cv2.drawMatches(query_img, kp1,
                         train_img, kp2,
                         good_matches, None,
                         flags=2)

plt.figure(figsize=(12,6))
plt.imshow(result)
plt.title("Correspondances SIFT")
plt.show()
