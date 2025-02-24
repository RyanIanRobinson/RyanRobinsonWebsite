// Function to dynamically load external HTML files (header & footer)
function loadComponent(id, file) {
    fetch(file)
        .then(response => response.text())
        .then(data => document.getElementById(id).innerHTML = data)
        .catch(error => console.error(`Error loading ${file}:`, error));
}

// Load header and footer
document.addEventListener("DOMContentLoaded", function () {
    loadComponent("header-container", "/includes/header.html");
    loadComponent("footer-container", "/includes/footer.html");
});

// Hardcoded script contents
const scriptContents = {
    
    "captioning": `
import torch
from diffusers import StableDiffusionPipeline
import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration, GPT2Tokenizer, GPTNeoForCausalLM
import os
import logging
# from dalle_mini import DalleBart, DalleBartProcessor
import clip

image_model_name = "ethzanalytics/blip2-flan-t5-xl-sharded"
gpt_model_name = "EleutherAI/gpt-neo-2.7B"
photo_generator_model_name_1 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
photo_generator_model_name_2 = "dalle-mini/dalle-mini/mega-1-fp16:latest"
photo_generator_model_name_3 = "ViT-B/32"

logging.basicConfig(level=logging.DEBUG)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))

def create_caches(model_name):
    folder_tokenizer_cache = os.path.join(script_dir, "tokenizer_cache", model_name)  # Cache folder
    folder_processor_cache = os.path.join(script_dir, "processor_cache", model_name)  # Cache folder
    folder_model_cache = os.path.join(script_dir, "model_cache", model_name)  # Cache folder

    os.makedirs(folder_tokenizer_cache, exist_ok = True)
    os.makedirs(folder_processor_cache, exist_ok = True)
    os.makedirs(folder_model_cache, exist_ok = True)

    print(f"Created caches for {model_name}")

    return folder_tokenizer_cache, folder_processor_cache, folder_model_cache

# Function to get granular description using VILT
def get_granular_description(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Process the image with VILT's processor
    inputs = image_processor(image, return_tensors="pt", prompt=prompt_image_description, padding=True, truncation=True)

    output = image_model.generate(**inputs)
    description = image_processor.decode(output[0], skip_special_tokens=True)

    return description

# Function to generate a funny summary using GPT-Neo
def generate_funny_summary(description):

    # Create a funny prompt for the description
    prompt = f"Rewrite the following caption as a funny, absurd, and unexpected scenario: \"{description}\""

    # Encode the prompt
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt")

    # Generate funny summary
    output_ids = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.8)

    # Decode the generated output
    funny_summary = gpt_tokenizer.decode(output_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)

    return funny_summary

###########################################################################
########## BLIP IMAGE TO TEXT SUMMARY
###########################################################################

image_folder_tokenizer_cache, image_folder_processor_cache, image_folder_model_cache = create_caches(image_model_name)
image_processor = BlipProcessor.from_pretrained(image_model_name, cache_dir=image_folder_processor_cache)
image_model = Blip2ForConditionalGeneration.from_pretrained(image_model_name, cache_dir=image_folder_model_cache)

# Test image URL
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
img_path = "demo_image.jpg"
img_data = requests.get(img_url).content
with open(img_path, 'wb') as f:
    f.write(img_data)

prompt_image_description = "Please describe this image with as many details as possible, including people, objects, actions, background, and any additional sensory details."

granular_description = get_granular_description(img_path)
print(f"\n\nGranular Description: {granular_description}\n\n")

###########################################################################
########## GPT TEXT SUMMARY TO FUNNY HYPOTHETICAL
###########################################################################

gpt_folder_tokenizer_cache, gpt_folder_processor_cache, gpt_folder_model_cache = create_caches(gpt_model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name, cache_dir=gpt_folder_tokenizer_cache)
print(f"Downloaded processor for {gpt_model_name}")
gpt_model = GPTNeoForCausalLM.from_pretrained(gpt_model_name, cache_dir=gpt_folder_model_cache)
print(f"Downloaded processor for {gpt_model_name}")

# Generate funny summary using GPT-Neo
funny_summary = generate_funny_summary(granular_description)
print(f"\n\nFunny Summary: {funny_summary}\n\n")

###########################################################################
########## FUNNY HYPOTHETICAL TO IMAGE
###########################################################################

########## Stable Diffusion
image_folder_tokenizer_cache, image_folder_processor_cache, image_folder_model_cache = create_caches(photo_generator_model_name_1)
pipe = StableDiffusionPipeline.from_pretrained(photo_generator_model_name_1, cache_dir=image_folder_model_cache, torch_dtype=torch.float16)
pipe.to("cuda")

# Generate image based on prompt
prompt = funny_summary
image = pipe(prompt).images[0]

image.show()

########## DALLE

# Load the processor and model
image_folder_tokenizer_cache, image_folder_processor_cache, image_folder_model_cache = create_caches(photo_generator_model_name_2)
processor = DalleBartProcessor.from_pretrained(photo_generator_model_name_2, cache_dir=image_folder_processor_cache)
model = DalleBart.from_pretrained(photo_generator_model_name_2, cache_dir=image_folder_model_cache)

# Generate an image based on a prompt
prompt = funny_summary
inputs = processor([prompt], return_tensors="pt", padding=True)

outputs = model.generate(**inputs)
image = Image.fromarray(outputs[0])
image.show()

############ CLIP
# Load the CLIP model
image_folder_tokenizer_cache, image_folder_processor_cache, image_folder_model_cache = create_caches(photo_generator_model_name_3)
model, preprocess = clip.load(photo_generator_model_name_3, cache_dir=image_folder_model_cache, device="cuda")

# Load and preprocess image
image = Image.open(img_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

# Generate textual description for the image
text_input = clip.tokenize([funny_summary]).to(device)

# Get image and text features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)

# Compute similarity
similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()
`
,
            "digital_photoframe_functions": 
`
##########################################################################
`
,
            "digital_photoframe_main": 
`
print("importing libraries: ", end = "")
import os
print("os", end = "")
import time
print(", time", end = "")
from PIL import Image
print(", PIL", end = "")
from transformers import Blip2Processor, Blip2ForConditionalGeneration
print(", transformers (Blip2Processor and Blip2ForConditionalGeneration)")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and processor
print("Loading processor and model...")
folder_processor_cache = os.path.join(script_dir, "processor_cache") # Subfolder in the current directory
folder_model_cache = os.path.join(script_dir, "model_cache")  # Subfolder in the current directory
print("Created cache directories")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = folder_processor_cache)
print("Loaded processor")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = folder_model_cache)
print("Loaded model")

# Define the directory containing the photos
photos_dir = os.path.join(script_dir, "photos")

# Define the prompt
prompt = """
Imagine you are a dark humour comedian.
Write a hilarious caption that mocks the person(s) in the photo.
A caption should address the people in the photo, the setting, talk in first person, and be 5-10 words.

The following are example captions, that must not be used, and the relevent context, I'd find funny:
1. 'Baby, the destroyer of happiness.'; A mother crying with a baby smiling.
2. 'The face I make when no one knows my evil master plan.'; A close up of a baby smiling playing with lego.
3. 'I hate when I have to take my dad for walks'; A baby crying while being pushed in a pram by a dad.
4. 'I wish I took the other baby at the hospital'; An adult and baby crying.
5. 'Little did I know this would be my biggest regret'; A newly wed couple.
"""

# Loop through all photos in the directory indefinitely
while True:
    for photo_filename in os.listdir(photos_dir):
        # Create full path to the photo
        photo_path = os.path.join(photos_dir, photo_filename)

        # Skip non-image files
        if not (photo_path.lower().endswith((".jpg", ".jpeg", ".png"))):
            continue

        # Process each image
        print(f"Processing photo: {photo_filename}...")
        try:
            image = Image.open(photo_path).convert("RGB")

            # Prepare inputs
            inputs = processor(images=image, text=prompt, return_tensors="pt")

            # Generate caption
            outputs = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.9)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Display the caption
            print(f"Caption for {photo_filename}: {caption}\n")

            # Wait for 15 seconds before moving to the next image
            time.sleep(15)

        except Exception as e:
            print(f"Error processing {photo_filename}: {e}")

`
,
            "digital_photoframe_params": 
`
##########################################################################
`
,
            "recognising_faces": 
`
from datetime import datetime
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

def cluster_embeddings(embeddings, name):
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

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plot_directory = os.path.join(script_dir, "cluster_plots")
    plot_filename = f"{name}_{timestamp}.png"
    
    # Plot the 2D PCA projection for visual inspection
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title("2D PCA visualization of embeddings")
    plt.savefig(os.path.join(plot_directory, plot_filename))
    plt.close() 

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
        clusters = cluster_embeddings(embeddings, name)
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
`
,
            "testing": 
`
print("importing libraries: ", end = "")
import os
print("os", end = "")
import time
print(", time", end = "")
from PIL import Image, ImageTk, ImageDraw, ImageFont
print(", PIL", end = "")
import tkinter as tk
print(", tkinter", end = "")
import torch
print(", torch", end = "")
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, GPT2Tokenizer, pipeline, logging
logging.set_verbosity_error()
print(", transformers. All done :)")

import warnings
warnings.filterwarnings("ignore")

# Setup models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load BLIP2 processor and model
folder_processor_cache = os.path.join(script_dir, "processor_cache")  # Cache folder
folder_model_cache = os.path.join(script_dir, "model_cache")  # Cache folder
folder_tokenizer_cache = os.path.join(script_dir, "tokenizer_cache")  # Cache folder
print("Loading BLIP2 tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_tokenizer_cache)
print("Loading BLIP2 processor...")
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", tokenizer=tokenizer, cache_dir=folder_processor_cache)
print("Loading BLIP2 model...")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_model_cache, ignore_mismatched_sizes=True).to(device)

# Load GPT2 tokenizer and text generation pipeline
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text_generator = pipeline("text-generation", model="gpt2", tokenizer=gpt2_tokenizer)

# Open image?
def open_image(image_path):
    try:
        # Open the image using Pillow
        image = Image.open(image_path)
                    
        # Set up Tkinter window
        root = tk.Tk()
        root.title("Image Viewer")

        # Convert the image to a format Tkinter can display
        photo = ImageTk.PhotoImage(image)

        # Create a label widget to display the image
        label = tk.Label(root, image=photo)
        label.pack()

        # Ensure the image is not garbage collected by keeping a reference
        label.image = photo

        # Function to close the window after 15 seconds
        def close_window():
            root.quit()
            root.destroy()

        # Wait for 15 seconds before closing the window
        root.after(5000, close_window)

        # Start the Tkinter main event loop
        root.mainloop()

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Function to generate summary from image
def generate_summary(image_path, prompt):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        summary = processor.decode(out[0], skip_special_tokens=True)
        return summary if summary else "No summary generated."
    except Exception as e:
        print(f"Error generating summary for {image_path}: {e}")
        return "Error generating summary."

# Function to generate hypothetical scenario based on prompt
def generate_hypothetical(scenario_prompt):
    return text_generator(scenario_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

# Function to display image with overlaid text for 5 seconds
def display_image_with_text(image_path, text, display_time=5):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Default font

    # Use textbbox instead of textsize in modern Pillow versions
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
    draw.text(text_position, text, font=font, fill=(255, 255, 255))  # White text

    image.show()
    time.sleep(display_time)
    image.close()

# Function to display hypothetical scenario image
def display_hypothetical_image(hypothetical_text, display_time=5):
    generated_image = generate_image_from_text(hypothetical_text)
    generated_image.show()
    time.sleep(display_time)
    generated_image.close()

# Placeholder for hypothetical image generation function (using DALL-E or similar)
def generate_image_from_text(hypothetical_text):
    # This should be replaced with actual image generation API (DALL-E, Stable Diffusion, etc.)
    # For now, we return a simple image with the hypothetical text written on it
    width, height = 300, 200
    image = Image.new("RGB", (width, height), (73, 109, 137))  # Placeholder background
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), hypothetical_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(text_position, hypothetical_text, font=font, fill=(255, 255, 255))  # White text
    return image

# Function to handle displaying image, summary, and hypothetical scenario
def process_images_in_folder(folder_path):
    
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        print(image_path)
        
        # Skip non-image files
        if not (image_path.lower().endswith((".jpg", ".jpeg", ".png"))):
            continue

        # Process each image
        print(f"Processing photo: {image_path}...")

        # open_image(image_path)

        # Generate summary of the image
        summary = generate_summary(image_path, prompt)
        print(f"Summary for {image_file}: {summary}")

        # Display image with summary for 5 seconds
        display_image_with_text(image_path, summary, 5)

        # Generate hypothetical situation
        hypothetical = generate_hypothetical(summary)
        print(f"Hypothetical for {image_file}: {hypothetical}")

        # Display hypothetical message for 5 seconds
        display_image_with_text(image_path, hypothetical, 5)

        # Display hypothetical generated image for 5 seconds
        display_hypothetical_image(hypothetical, 5)

        # Move to the next image
        print(f"Moving to next image...\n")
        
# Main execution
photos_dir = os.path.join(script_dir, "photos") # Path to your folder with images
print("photos_dir: ", photos_dir)

prompt = "Write a hilarious caption that mocks the person(s) in the photo. A caption should address the people in the photo, the setting, talk in first person, and be 5-10 words."

prompt2 = """
Imagine you are a dark humour comedian.
Write a hilarious caption that mocks the person(s) in the photo.
A caption should address the people in the photo, the setting, talk in first person, and be 5-10 words.

The following are example captions, that must not be used, and the relevent context, I'd find funny:
1. 'Baby, the destroyer of happiness.'; A mother crying with a baby smiling.
2. 'The face I make when no one knows my evil master plan.'; A close up of a baby smiling playing with lego.
3. 'I hate when I have to take my dad for walks'; A baby crying while being pushed in a pram by a dad.
4. 'I wish I took the other baby at the hospital'; An adult and baby crying.
5. 'Little did I know this would be my biggest regret'; A newly wed couple.
"""

process_images_in_folder(photos_dir)

`
};

// Function to display script content
function showCode(scriptName) {
    const codeDisplay = document.getElementById('code-content');
    codeDisplay.textContent = scriptContents[scriptName] || "Error: Script content not found.";
}
