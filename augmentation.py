import cv2
import numpy as np
import os

# Charger la liste des images depuis le répertoire
images = os.listdir('C:\\Users\\chaima\\Desktop\\DossierTp\\moteur-de-recherche\\BE')
for i,image_name in enumerate(images):
    img = cv2.imread(f'C:\\Users\\chaima\\Desktop\\DossierTp\\moteur-de-recherche\\BE\\{image_name}')
    # Rotation de 30 degrés
    rows, cols,_ = img.shape
    M_rotation = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
    img_rotation = cv2.warpAffine(img, M_rotation, (cols, rows))
    new_image_name = f"{i+ 51}.jpg"
    new_image_path = os.path.join('C:\\Users\\chaima\\Desktop\\DossierTp\\moteur-de-recherche\\BE', new_image_name)
    cv2.imwrite(new_image_path, img_rotation)
    # Translation de 50 pixels vers la droite et 20 pixels vers le bas
    M_translation = np.float32([[1, 0, 50], [0, 1, 20]])
    img_translation = cv2.warpAffine(img, M_translation, (cols, rows))
    new_image_name = f"{i+ 52}.jpg"
    new_image_path = os.path.join('C:\\Users\\chaima\\Desktop\\DossierTp\\moteur-de-recherche\\BE', new_image_name)
    cv2.imwrite(new_image_path, img_translation)
    # Zoom avec une matrice d'échelle
    M_zoom = np.float32([[1.5, 0, 0], [0, 1.5, 0]])
    img_zoom = cv2.warpAffine(img, M_zoom, (cols, rows))
    new_image_name = f"{i+ 53}.jpg"
    new_image_path = os.path.join('C:\\Users\\chaima\\Desktop\\DossierTp\\moteur-de-recherche\\BE', new_image_name)
    cv2.imwrite(new_image_path, img_zoom)
    