"""
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from torchvision import transforms

# Ensure you load the model and processor correctly
folder_processor_cache = "F:/GitHub/Websites/Projects/digital_photoframe/processor_cache"
folder_model_cache = "F:/GitHub/Websites/Projects/digital_photoframe/model_cache"
folder_tokenizer_cache = "F:/GitHub/Websites/Projects/digital_photoframe/tokenizer_cache"

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_processor_cache)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_model_cache)

# Open and process the image
raw_image = Image.open("F:/GitHub/Websites/Projects/digital_photoframe/photos/CD 1.jpg").convert('RGB')

# Get the input size compatible with the model
model_input_size = processor.image_processor.size
print("\n\nmodel_input_size: ", model_input_size)

# Extract the height and width from the dictionary
height = model_input_size['height']
width = model_input_size['width']

# Resize the image to the model's expected input size
raw_image = raw_image.resize((width, height))
print("\n\nresized raw_image: ", raw_image.size)

# Question to ask
question = "How many dogs are in the picture?"

# Use the processor to handle both image and question properly
inputs = processor(raw_image, question, return_tensors="pt")
print(f"\n\nImage tensor shape: {inputs['pixel_values'].shape}")
print(f"Text tensor shape: {inputs['input_ids'].shape}")
print("Pixel values (image tensor):", inputs['pixel_values'])
print("Input IDs (text tokens):", inputs['input_ids'])

print("HOLY COW!!!!", inputs)

##################################################################################################################################################################################
# Generate the output
print("Giddy up")
# Pass the inputs to the model for generation
try:
    out = model.generate(
        pixel_values=inputs['pixel_values'],  # Image tensor
        input_ids=inputs['input_ids'],        # Text tensor
        attention_mask=inputs['attention_mask']  # Attention mask if available
    )
    
    print("Generated Output:", out)
    decoded_output = processor.decode(out[0], skip_special_tokens=True)
    print("Decoded Output:", decoded_output)
    
except Exception as e:
    print(f"An error occurred: {e}")
print("Giddy down")
##################################################################################################################################################################################

try:
    out = model.generate(**inputs)
    print("Output:", out)
    print("Decoded Output:", processor.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print(f"An error occurred: {e}\n\n")
    print("Inputs:\n", inputs, "\n\n")
"""

import torch
import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

folder_processor_cache = "F:/GitHub/Websites/Projects/digital_photoframe/processor_cache"
folder_model_cache = "F:/GitHub/Websites/Projects/digital_photoframe/model_cache"
folder_tokenizer_cache = "F:/GitHub/Websites/Projects/digital_photoframe/tokenizer_cache"

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=folder_processor_cache)
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    cache_dir=folder_model_cache,
    torch_dtype=torch.float32,
    device_map="auto",
    quantization_config=quantization_config
)
model_input_size = model.config.vision_config.image_size
print(f"Model input size: {model_input_size}")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Ensure the image was loaded properly
print(f"Raw image size: {raw_image.size}")  # Prints image dimensions

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt", padding=True, truncation=True).to("cuda", torch.float32)

# Print the shapes of the tensors to ensure they are correct
print(f"Image tensor shape: {inputs['pixel_values'].shape}")
print(f"Text tensor shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")

# Check if any of the tensors have shape [0]
if inputs['pixel_values'].shape[0] == 0:
    print("Error: The image tensor has shape [0], meaning no data is present.")
if inputs['input_ids'].shape[0] == 0:
    print("Error: The text tensor has shape [0], meaning no input tokens were processed.")
if inputs['attention_mask'].shape[0] == 0:
    print("Error: The attention mask tensor has shape [0], meaning no attention mask was created.")

print(f"Input Image Tensor: {inputs['pixel_values'].shape}")
print(f"Input Text Tensor: {inputs['input_ids'].shape}")
print(f"Input Attention Mask Tensor: {inputs['attention_mask'].shape}")
print("Processor output keys:", inputs.keys())
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).major)
print(torch.cuda.get_device_properties(0).minor)
print(model.hf_device_map)

# Run the model to generate the output
try:
    out = model.generate(**inputs)
    print(f"Model output: {out}")
except Exception as e:
    print(f"An error occurred during generation: {e}")

# Decode the output
try:
    decoded_output = processor.decode(out[0], skip_special_tokens=True)
    print(f"Decoded output: {decoded_output}")
except Exception as e:
    print(f"An error occurred during decoding: {e}")