# lambda_function.py
import numpy as np
from PIL import Image
from urllib.request import urlopen
import onnxruntime as ort
from torchvision import transforms

# Load the ONNX model once when the container starts
ort_session = ort.InferenceSession("hair_classifier_empty.onnx")  # model already in container
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Preprocessing transforms
preprocess_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Utility functions
def download_image(url):
    with urlopen(url) as resp:
        img = Image.open(resp)
        return img

def prepare_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((200, 200), Image.NEAREST)
    return img

def preprocess_image(img):
    x = preprocess_transform(img)
    x = x.unsqueeze(0).numpy()
    return x

# Lambda handler
def handler(event, context):
    url = event["image_url"]
    img = download_image(url)
    img = prepare_image(img)
    x = preprocess_image(img)
    pred = ort_session.run([output_name], {input_name: x})[0]
    return float(pred[0][0])
