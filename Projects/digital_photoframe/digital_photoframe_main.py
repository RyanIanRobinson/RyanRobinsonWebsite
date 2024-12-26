print("importing libraries: ", end = "")
import os
print("os", end = "")
import time
print(", time", end = "")
from PIL import Image
print(", PIL", end = "")
from transformers import Blip2Processor, Blip2ForConditionalGeneration
print(", transformers (Blip2Processor and Blip2ForConditionalGeneration)")

# Load model and processor
print("Loading processor and model...")
folder_processor_cache = os.path.join(os.getcwd(), "processor_cache") # Subfolder in the current directory
folder_model_cache = os.path.join(os.getcwd(), "model_cache")  # Subfolder in the current directory
print("Created cache directories")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = folder_processor_cache)
print("Loaded processor")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = folder_model_cache)
print("Loaded model")

# Define the directory containing the photos
script_dir = os.path.dirname(os.path.abspath(__file__))
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
