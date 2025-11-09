import modal 
import torch
import numpy as np
import requests

import base64

from torchvision import transforms

from PIL import Image

import io

from pydantic import BaseModel

from model import ResNet34



app = modal.App("catdog-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("catdog-model")


class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),      # igualamos tamaños
        transforms.ToTensor(),               # pasa a tensor (C x H x W)
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalizar valores entre -1 y 1 
        ])

    def process_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # -> (1, C, H, W)
    
    
class InferenceRequest(BaseModel):
    image_data: str


@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)

class ImageClassifier:

    @modal.enter()

    def load_model(self):

        print("Loading model on enter")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load('/models/best_model.pth', map_location=self.device)
        self.classes = checkpoint['classes']

        self.model = ResNet34(num_classes=len(self.classes))   
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.image_processor = ImageProcessor()

        print("Model loaded")


    @modal.fastapi_endpoint(method="POST")

    def inference(self, request: InferenceRequest):
        # 1. Base64 → bytes
        image_bytes = base64.b64decode(request.image_data)

        # 2. Procesar imagen
        image_tensor = self.image_processor.process_image(image_bytes).to(self.device)

        # 3. Inferencia
        with torch.no_grad():
            output, feature_maps = self.model(image_tensor, return_feature_maps = True)
            probabilities = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities[0], 2)

        # 4. Formatear respuesta
        predictions = [
            {"class": self.classes[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]


        viz_data = {}

        for name, tensor in feature_maps.items():

            if tensor.dim() == 4:  # [batch, channels, height, width]
                # Promedio de canales → 2D activación
                activation_map = torch.mean(tensor, dim=1).squeeze(0)
                numpy_array = activation_map.cpu().numpy()
                clean_array = np.nan_to_num(numpy_array)

                viz_data[name] = {
                    "shape": list(clean_array.shape),
                    "values": clean_array.tolist()
                }

        
        response = {
            "predictions": predictions,
            "visualization": viz_data
        }


        return response


@app.local_entrypoint()

def main():
    # 1. Leer imagen y convertir a base64
    with open("gatete.webp", "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image_data": image_b64}

    # 2. Crear instancia local y obtener URL del endpoint
    server = ImageClassifier()
    url = server.inference.get_web_url()

    # 3. Enviar POST
    response = requests.post(url, json=payload)
    response.raise_for_status()

    # 4. Interpretar resultado
    result = response.json()

    print("\nTop predictions:")
    for pred in result.get("predictions", []):
        print(f"  - {pred['class']}  {pred['confidence']:.2%}")

       
