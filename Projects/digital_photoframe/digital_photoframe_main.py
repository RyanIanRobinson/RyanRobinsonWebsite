print("Get libraries")
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# Load model and processor
print("Load model and processor...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
print("model")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Load an image
print("Load an image...")
image = Image.open("C:/Users/ryanr/Desktop/Alice/7.jpg").convert("RGB")

# Define your prompt
print("Define your prompt...")
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

# Prepare inputs
print("Prepare inputs...")
inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate caption
print("Generate caption...")
print("Outputs...")
outputs = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.9)
print("Caption...")
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Caption:", caption)

