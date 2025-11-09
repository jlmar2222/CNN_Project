import torch
import numpy as np
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import io
from model import ResNet34


# ---- Configuración ----
MODEL_PATH = "best_model.pth"
IMAGE_PATH = "perrete.webp"   # Cambia esto por la imagen que quieras
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Procesamiento de imagen ----
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    return tensor, image


# ---- Carga del modelo ----
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    classes = checkpoint.get("classes", ["dogs", "cats"])  # fallback
    model = ResNet34(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"✅ Modelo cargado correctamente ({len(classes)} clases)")
    return model, classes


# ---- Inferencia ----
def predict(model, image_tensor, classes):
    with torch.no_grad():
        output, feature_maps = model(image_tensor.to(DEVICE), return_feature_maps=True)
        probs = torch.softmax(output, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    pred_class = classes[top_idx.item()]
    confidence = top_prob.item()

    print(f"\n🐱🐶 Predicción: {pred_class} ({confidence:.2%})")

    return pred_class, confidence, feature_maps

def visualize_feature_maps(feature_maps, max_channels=8, cmap="plasma"):
    """
    Visualiza los feature maps del último bloque de cada layer
    (tanto antes como después de la ReLU), más conv1.
    """
    layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    
    plt.figure(figsize=(16, 12))
    plot_idx = 1

    for layer in layers:
        if layer == "conv1":
            fmap = feature_maps[layer]
            fmap = fmap.squeeze(0)
            fmap_subset = fmap[:max_channels].cpu().numpy()
            
            for i in range(fmap_subset.shape[0]):
                plt.subplot(len(layers), max_channels, plot_idx)
                plt.imshow(fmap_subset[i], cmap=cmap)  # ← usa el argumento
                plt.axis("off")
                if i == 0:
                    plt.ylabel(layer, fontsize=10)
                plot_idx += 1
            continue
        
        block_names = [k for k in feature_maps.keys() if k.startswith(layer)]
        last_block_prefixes = sorted(set([".".join(k.split(".")[:2]) for k in block_names]))[-1]
        
        for kind in ["conv", "relu"]:
            key = f"{last_block_prefixes}.{kind}"
            if key not in feature_maps:
                continue
            fmap = feature_maps[key].squeeze(0)
            fmap_subset = fmap[:max_channels].cpu().numpy()

            for i in range(fmap_subset.shape[0]):
                plt.subplot(len(layers)*2, max_channels, plot_idx)
                plt.imshow(fmap_subset[i], cmap=cmap)  # ← usa el argumento
                plt.axis("off")
                if i == 0:
                    plt.ylabel(f"{layer}\n({kind})", fontsize=10)
                plot_idx += 1

    plt.tight_layout()
    plt.show()


# ---- Ejecución principal ----
if __name__ == "__main__":
    # 1. Preprocesar imagen
    image_tensor, raw_image = preprocess_image(IMAGE_PATH)

    # 2. Cargar modelo
    model, classes = load_model()

    # 3. Inferencia
    pred_class, confidence, feature_maps = predict(model, image_tensor, classes)

    # 4. Mostrar imagen original
    plt.imshow(raw_image)
    plt.title(f"Pred: {pred_class} ({confidence:.2%})")
    plt.axis("off")
    plt.show()

    # 5. Visualizar feature maps
    visualize_feature_maps(feature_maps, cmap="plasma")
