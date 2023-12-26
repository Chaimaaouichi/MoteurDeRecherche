import cv2
import numpy as np
from scipy.spatial.distance import cdist
import os 
import json
import tqdm
def calculate_histogram(image_path, N):

    image =cv2.imread(image_path)
    # Diviserchaque axe de l'espace de couleurs en N intervalles
    hist_bins = [N, N, N]

    try:
        histogram = cv2.calcHist([image], [0, 1, 2], None, hist_bins,  [0, 256, 0, 256, 0, 256])
    except cv2.error as e:
        print(f"Error calculating histogram: {e}")
        return None

    # Normaliser l'histogramme
    histogram = cv2.normalize(histogram, histogram).flatten().tolist()

    return histogram

def euclidean_distance(hist1,hist2):
  distance = np.sqrt(np.sum(np.square(np.array(hist1) - np.array(hist2))))
  return distance
def calcul_similarité_histogramme(img_requete_path,json_file):
  histogram_requete = calculate_histogram(img_requete_path,8)
  with open(json_file, 'r') as f:
        data = json.load(f)
  distances = []
  for entry in data:
        image_name = entry["image_path"]
        histogram = entry["histogram"]
        # Calculer la distance eucliedienne entre les histogrammes
        distance = euclidean_distance(histogram_requete, histogram)
        distances.append((distance,image_name))
  distances.sort()
  top3 =distances[:3]
  return top3

def texture_descriptor(image_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculer la Transformée de Fourier 2D
    fft_image = np.fft.fft2(image)

    # Obtenir le spectre en amplitude
    amplitude_spectrum = np.abs(fft_image)

    # Découper la moitié supérieure en 6x3 blocs
    blocks = np.array_split(amplitude_spectrum[:amplitude_spectrum.shape[0]//2, :], 6, axis=0)
    blocks = [np.array_split(block, 3, axis=1) for block in blocks]

    # Calculer le logarithme de l'énergie moyenne sur chaque bloc
    descriptors = [np.log(np.mean(np.square(block))) for row in blocks for block in row]

    return descriptors
def manhattan_distance(descriptor1, descriptor2):
    # Calcul de la distance de Manhattan entre deux descripteurs
    return np.sum(np.abs(np.array(descriptor1) - np.array(descriptor2)))
def calcul_similarité_texture(img_requete_path,json_file):
  descriptors=texture_descriptor(img_requete_path)
  with open(json_file, 'r') as f:
        data = json.load(f)
  distances = []
  for entry in data:
        image_name = entry["filename"]
        descriptor = entry["descriptors"]
        # Calculer la distance de Manhattan entre les descripteurs
        distance = manhattan_distance(descriptors, descriptor)

        # Ajouter la distance et le nom de l'image à la liste
        distances.append((distance, image_name))

  distances.sort()

    # Récupérer les trois images les plus similaires
  top3_similar_images = distances[:3]

  return top3_similar_images
def normalize_descriptor(descriptor):
    norm = np.linalg.norm(descriptor)
    if norm == 0:
        return descriptor
    return descriptor / norm
def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)
def Construction_Caracteristicsvector(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    dst1 = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(dst1, None)
    keypoints = np.argwhere(harris_corners > 0.01 * harris_corners.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in keypoints]
    _, descriptors = sift.compute(image, keypoints)
    normalized_descriptors = np.array([normalize_descriptor(descriptor) for descriptor in descriptors])
    return normalized_descriptors
def calculate_similarity_2IMAGES(image1, image2):
    vector1 = Construction_Caracteristicsvector(image1)
    vector2 = Construction_Caracteristicsvector(image2)
    S = cdist(vector1,vector2,'euclidean')
    firstmatch = np.argmin(S, axis=1)
    secondmatch= np.argmin(S, axis=0)
    reciprocal_matches = np.column_stack((firstmatch[secondmatch], np.arange(len(firstmatch))))
    similarity_score= reciprocal_matches.shape[0]

    return S,similarity_score
def Local_Descriptors_Search(image_path):
    images_directory = "C:\Users\chaima\Desktop\DossierTp\moteur-de-recherche\BE"
    image_files = os.listdir(images_directory)
    sift_descriptors = {}
    for image_file in image_files:
      image_path = os.path.join(images_directory, image_file)
      kp, des = Construction_Caracteristicsvector(image_path)
      sift_descriptors[image_file] = des
    num_images = len(image_files)
    similarity_matrix = np.zeros((num_images, num_images))
    for i,image1 in tqdm(range(num_images), desc="Calcul de la matrice de similarité"):
      for j,image2 in range(i+1, num_images):
        image1_path = os.path.join(images_directory, image1)
        image2_path = os.path.join(images_directory, image1)
        _,score=calculate_similarity_2IMAGES(image1_path,image2_path)
        # Remplir la matrice de similarité avec le nombre de correspondances réciproques
        similarity_matrix[i, j] = similarity_matrix[j, i] = len(score)
    mean_value = similarity_matrix.mean()
    std_dev = similarity_matrix.std()
    normalized_similarity_matrix = (similarity_matrix - mean_value) / std_dev
    return (normalized_similarity_matrix)






    
