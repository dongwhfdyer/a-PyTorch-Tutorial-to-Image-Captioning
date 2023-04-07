
#---------kkuhn-block------------------------------ # test2
import os.path

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LOCAL_CLIP_FILE = "pretrained/clip-vit-base-patch32.pt"

CLIPPROCESSOR_NAME = "openai/clip-vit-base-patch32"
LOCAL_CLIPPROCESSOR_FILE = "pretrained/clip_processor.pt"

def load_huggingface_model(model_class,model_name, local_file):
    if os.path.exists(local_file):
        print(f"Loading {model_class} from local file {local_file}")
        model = model_class.from_pretrained(local_file)
    else:
        print(f"Loading {model_class} from huggingface model {model_name}")
        model = model_class.from_pretrained(model_name)
    return model

model = load_huggingface_model(CLIPModel, CLIP_MODEL_NAME, LOCAL_CLIP_FILE)
processor = load_huggingface_model(CLIPProcessor, CLIPPROCESSOR_NAME, LOCAL_CLIPPROCESSOR_FILE)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
#---------kkuhn-block------------------------------


print("--------------------------------------------------")
