from mtcnn import MTCNN
import tempfile
import torch
import requests
from PIL import Image
import os
import warnings
from transformers import CLIPProcessor, CLIPModel
import clip
import cv2

# Suppress all warnings
warnings.filterwarnings('ignore')

model_name = "openai/clip-vit-base-patch16"
script_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
"""
folder_tokenizer_cache = os.path.join(script_dir, "tokenizer_cache", model_name)  # Cache folder
folder_processor_cache = os.path.join(script_dir, "processor_cache", model_name)  # Cache folder
folder_model_cache = os.path.join(script_dir, "model_cache", model_name)  # Cache folder

os.makedirs(folder_tokenizer_cache, exist_ok = True)
os.makedirs(folder_processor_cache, exist_ok = True)
os.makedirs(folder_model_cache, exist_ok = True)

#tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=folder_tokenizer_cache)
processor = CLIPProcessor.from_pretrained(model_name, cache_dir=folder_processor_cache)
model = CLIPModel.from_pretrained(model_name, cache_dir=folder_model_cache)

# Load test image
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Define the text prompts
prompts = ["two cats", "a photo of a dog", "a photo of a human", "dog and human caricature"]

# Preprocess the image and the prompt
inputs = processor(text=prompts, images=raw_image, return_tensors="pt", padding=True)

# Run the model to generate the output
try:
    output = model(**inputs)
    logits_per_image = output.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    
    print(f"Model output: {output}")
    print(f"Probabilities: {probs}")
except Exception as e:
    print(f"An error occurred during generation: {e}")
"""
#######################################################################################################################################
# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
# Load OpenCV's pre-trained face detector (Haar Cascade)
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Function to calculate average embeddings for a set of images
def get_average_embedding(image_folder):
    embeddings = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        embedding = get_image_embedding(image_path)
        embeddings.append(embedding)
    return torch.mean(torch.stack(embeddings), dim=0)

# Function to calculate similarity scores (probabilities) for each category
def get_probabilities(face_embedding, reference_embeddings):
    similarities = {}
    for name, ref_embedding in reference_embeddings.items():
        similarity = torch.cosine_similarity(face_embedding, ref_embedding)
        similarities[name] = similarity.item()  # Store cosine similarity as probability
    
    return similarities

# Function to detect faces in an image using OpenCV
"""
def detect_faces(image_path):
    # OpenCV face detection
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
"""
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
        
        # Only accept matches above the threshold
        if best_match_prob >= threshold:
            results.append({
                "best_match": best_match_name,
                "probabilities": probability
            })
        else:
            results.append({
                "best_match": "Did not meet threshold",
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
        
        print()  # Print a blank line between images

# Image folders for training data
grayson_folder = os.path.join(script_dir, "training_data", "grayson")
hayden_folder = os.path.join(script_dir, "training_data", "hayden")
josie_folder = os.path.join(script_dir, "training_data", "josie")
ryan_folder = os.path.join(script_dir, "training_data", "ryan")
tanto_folder = os.path.join(script_dir, "training_data", "tanto")
strangers_folder = os.path.join(script_dir, "training_data", "strangers")

# Create reference embeddings for each category
grayson_embedding = get_average_embedding(grayson_folder)
hayden_embedding = get_average_embedding(hayden_folder)
josie_embedding = get_average_embedding(josie_folder)
ryan_embedding = get_average_embedding(ryan_folder)
tanto_embedding = get_average_embedding(tanto_folder)
strangers_embedding = get_average_embedding(strangers_folder)

# Store reference embeddings in a dictionary
reference_embeddings = {
    "Grayson": grayson_embedding,
    "Hayden": hayden_embedding,
    "Josie": josie_embedding,
    "Ryan": ryan_embedding,
    "Tanto": tanto_embedding,
    "Strangers": strangers_embedding,
}

# Folder where you want to classify images
input_data_folder = os.path.join(script_dir, "input_data")
image_paths = [os.path.join(input_data_folder, img) for img in os.listdir(input_data_folder)]

detected_faces = []
for image_path in image_paths:
    result = classify_faces_in_image(image_path, reference_embeddings)
    detected_faces.append(result)

# Show the results
show_results(image_paths, detected_faces)