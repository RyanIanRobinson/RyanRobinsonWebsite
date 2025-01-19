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