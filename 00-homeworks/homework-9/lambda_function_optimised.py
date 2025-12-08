import numpy as np
from PIL import Image
from urllib.request import urlopen
import onnxruntime as ort

# Load the ONNX model once when the container starts
ort_session = ort.InferenceSession("hair_classifier_empty.onnx")  # model already in container
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Utility functions
def download_image(url):
    with urlopen(url) as resp:
        img = Image.open(resp)
        return img

def preprocess_image(img):
    # Ensure RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize
    img = img.resize((200, 200), Image.NEAREST)
    # Convert to NumPy array and scale to [0,1]
    x = np.array(img).astype(np.float32) / 255.0
    # Normalize per channel
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    # Change shape to (1, 3, H, W)
    x = np.transpose(x, (2, 0, 1))[np.newaxis, :, :, :]
    return x.astype(np.float32)  # ensure float32

# Lambda handler
def handler(event, context):
    url = event["image_url"]
    img = download_image(url)
    x = preprocess_image(img)
    pred = ort_session.run([output_name], {input_name: x})[0]
    return float(pred[0][0])
