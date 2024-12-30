import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import tempfile
import torch
from PIL import Image
import os
import warnings
import clip
import cv2
import numpy as np

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
detector = MTCNN()

# Function to process and encode images into CLIP embeddings
def get_image_embedding(image_input):
    # If the input is a file path (string), load the image
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            print(f"Invalid file path: {image_input}")
            return None
        image_input = Image.open(image_input)

    # Resize and preprocess the image for CLIP (224x224 resolution)
    image_input = preprocess(image_input).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
    
    return image_features / image_features.norm(dim=-1, keepdim=True)

def cluster_embeddings(embeddings):
    # Ensure embeddings is not empty
    if not embeddings:
        print("No embeddings provided.")
        return []

    # Convert embeddings to a 2D NumPy array
    embeddings_array = np.array([embedding.cpu().numpy().flatten() if isinstance(embedding, torch.Tensor) else np.array(embedding).flatten() for embedding in embeddings])
    
    # Check if the embeddings_array is empty
    if embeddings_array.size == 0:
        print("Embeddings array is empty after conversion.")
        print("Embeddings: ", embeddings)
        return []

    # Check if there are enough samples for PCA
    if embeddings_array.shape[0] < 2:
        print(f"Not enough samples for PCA. Number of samples: {embeddings_array.shape[0]}")
        print("Embeddings: ", embeddings)
        return []

    # Use PCA to reduce the dimensionality of the embeddings to 2D for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    # Plot the 2D PCA projection for visual inspection
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title("2D PCA visualization of embeddings")
    plt.show()

    # Apply HDBSCAN clustering with Euclidean distance
    clusterer = hdbscan.HDBSCAN(min_samples=3, metric='euclidean')
    clusterer.fit(reduced_embeddings)

    print("Clusterer: ", clusterer)
    print("Cluster labels: ", clusterer.labels_)

    return clusterer.labels_

def get_cluster_average(embeddings, labels):
    cluster_averages = {}
    for label in set(labels):
        clustered_embeddings = [embeddings[i] for i in range(len(embeddings)) if labels[i] == label]
        cluster_average = torch.mean(torch.stack(clustered_embeddings), dim=0)
        cluster_averages[label] = cluster_average
    print("cluster_averages: ", cluster_averages)
    return cluster_averages

# Function to calculate similarity scores (probabilities) for each category
def get_probabilities(face_embedding, reference_cluster_averages):
    similarities = {}
    for name, cluster_averages in reference_cluster_averages.items():
        # Compare the face_embedding to each cluster's average
        max_similarity = -1
        for cluster_label, cluster_avg in cluster_averages.items():
            # Use cosine similarity to compare face_embedding with each cluster's average
            similarity = torch.cosine_similarity(face_embedding, cluster_avg)
            max_similarity = max(max_similarity, similarity.item())

        similarities[name] = max_similarity
    
    return similarities

# Function to detect faces in an image
def detect_faces(image_path):
    image = cv2.imread(image_path)
    faces = detector.detect_faces(image)
    return [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]

# Function to classify individual faces in an image with a threshold (75% match)
def classify_faces_in_image(image_path, reference_embeddings, threshold=0.75):
    # Detect faces in the image
    faces = detect_faces(image_path)
    results = []  # Will store the results for each face in this image
    
    for (x, y, w, h) in faces:
        # Crop the face from the image
        image = Image.open(image_path)
        face_image = image.crop((x, y, x + w, y + h))
        
        # Create a temporary file to save the cropped face
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            face_image.save(temp_file_path)
        
        # Get the embedding for the cropped face
        face_embedding = get_image_embedding(temp_file_path)
        
        # Get the probabilities for each person in the reference embeddings
        probability = get_probabilities(face_embedding, reference_embeddings)
        
        # Find the best match by comparing probabilities
        best_match_name = None
        best_match_prob = 0.0
        for name, prob in probability.items():
            if prob > best_match_prob:
                best_match_prob = prob
                best_match_name = name
        
        print("best_match_name: ", best_match_name)
        print("best_match_prob: ", best_match_prob)

        # Append the result, including probabilities for all cases
        results.append({
            "best_match": best_match_name if best_match_prob >= threshold else "Did not meet threshold",
            "probabilities": probability
        })
        
        # Optionally, delete the temporary file after processing
        os.remove(temp_file_path)
    
    return results

# Function to show the results
def show_results(image_paths, detected_faces):
    print("\n\nResults:")
    for i, image_path in enumerate(image_paths):
        # Print image name (without path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Image: {image_name}")
        
        # Collect detected people
        detected_people = [result["best_match"] for result in detected_faces[i] if result["best_match"] != "Did not meet threshold"]

        print(f"Detected people: {', '.join(detected_people) if detected_people else 'None'}")
        
        # Print probabilities for each face
        for face_index, result in enumerate(detected_faces[i]):
            print(f"    Face {face_index + 1}: {result['best_match']}")
            for name, prob in result["probabilities"].items():
                print(f"        {name}: {prob:.18f}")
        
        print()

# Function to create folder paths and process embeddings and clusters
def process_person_data(names, base_dir):
    reference_embeddings = {}
    reference_cluster_averages = {}

    for name in names:
        print(f"\n\nname")
        # Create the folder path for each person dynamically
        folder = os.path.join(base_dir, name.lower())  # assuming folder names are lowercase
        if not os.path.exists(folder):
            print(f"Folder does not exist for {name}, skipping.")
            continue

        # Create reference embeddings for each category
        embeddings = [get_image_embedding(os.path.join(folder, img)) for img in os.listdir(folder)]
        
        # Store embeddings in reference_embeddings dictionary
        reference_embeddings[name] = embeddings
        print(f"reference_embeddings[name]: {name}", reference_embeddings[name][0][0])
        
        # Cluster and get average embeddings for each person
        clusters = cluster_embeddings(embeddings)
        cluster_averages = get_cluster_average(embeddings, clusters)
        
        # Store cluster averages in reference_cluster_averages dictionary
        reference_cluster_averages[name] = cluster_averages

    return reference_embeddings, reference_cluster_averages

# Base directory where the folders are located
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "training_data")

# List of names for people
names = ["Grayson", "Hayden", "Josie", "Ryan", "Tanto", "Strangers"]

# Process the data using the function
reference_embeddings, reference_cluster_averages = process_person_data(names, base_dir)
print("Person data has been processed")

# Folder where you want to classify images
input_data_folder = os.path.join(script_dir, "input_data")
image_paths = [os.path.join(input_data_folder, img) for img in os.listdir(input_data_folder)]

detected_faces = []
for image_path in image_paths:
    result = classify_faces_in_image(image_path, reference_cluster_averages)
    detected_faces.append(result)

# Show the results
show_results(image_paths, detected_faces)