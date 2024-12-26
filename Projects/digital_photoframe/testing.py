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

# Load BLIP2 processor and model
folder_processor_cache = os.path.join(os.getcwd(), "processor_cache")  # Cache folder
folder_model_cache = os.path.join(os.getcwd(), "model_cache")  # Cache folder
folder_tokenizer_cache = os.path.join(os.getcwd(), "tokenizer_cache")  # Cache folder
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
script_dir = os.path.dirname(os.path.abspath(__file__))
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
