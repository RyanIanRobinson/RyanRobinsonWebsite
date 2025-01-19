import os
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, GPT2Tokenizer, pipeline, logging
import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


folder_processor_cache = os.path.join(os.getcwd(), "processor_cache")  # Cache folder
folder_model_cache = os.path.join(os.getcwd(), "model_cache")  # Cache folder
folder_tokenizer_cache = os.path.join(os.getcwd(), "tokenizer_cache")  # Cache folder

print("Loading BLIP2 tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_tokenizer_cache)
print("Loading BLIP2 processor...")
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", tokenizer=tokenizer, cache_dir=folder_processor_cache)
print("Loading BLIP2 model...")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_model_cache, ignore_mismatched_sizes=True).to(device)

raw_image = Image.open("C:\Dragon\GitHub\Websites\Projects\digital_photoframe\photos\CD 1.jpg").convert("RGB")
inputs = processor(raw_image, return_tensors="pt").to(device)
out = model.generate(**inputs)

summary = processor.decode(out[0], skip_special_tokens=True)

# Print the generated summary
print(summary)
