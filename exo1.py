import cv2

img_path = "im1.jpg"
image_c = cv2.imread(img_path)
image = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

N = 25   # changer pour tester la boucle

g = image.copy()

for i in range(N):
    g = cv2.GaussianBlur(g, (3,3), 2, 0, cv2.BORDER_DEFAULT)


cv2.imshow('Blur image1',g)
cv2.waitKey(0)
cv2.imwrite("im1_blur_N.png", g)
cv2.waitKey(0)