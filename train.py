import torch
import torch.nn as nn

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

from torch.utils.data import Dataset , DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms 

import numpy as np 

from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim

from tqdm import tqdm

import modal

from model import ResNet34



app = modal.App("catdog-cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O catdog_db.zip",
             "cd /tmp && unzip catdog_db.zip",
             "mkdir -p /opt/catdog-data",
             "cp -r /tmp/cats_and_dogs_filtered/* /opt/catdog-data/",
             "rm -rf /tmp/catdog_db.zip /tmp/cats_and_dogs_filtered"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("catdog-data", create_if_missing=True)
model_volume = modal.Volume.from_name("catdog-model", create_if_missing=True)

class DogsCatsDataset(Dataset):
    def __init__(self, data_dir, split = 'train',transform=None):
        #self.data_dir = Path(data_dir)
        self.transform = transform

        if split == 'train':
            self.data_dir = Path(data_dir) / 'train'
        else:
            self.data_dir = Path(data_dir) / 'validation'


        self.img_paths = []
        self.labels = []
        self.classes = ["dogs", "cats"]

        # Recorrer subcarpetas
        for class_name in ["dogs", "cats"]:
            folder = self.data_dir / class_name
            for img_file in folder.iterdir():  # recorre cada imagen
                self.img_paths.append(img_file)

                # Asigna etiqueta según carpeta
                if class_name == "dogs":
                    self.labels.append(0)  # perro → 0
                else:
                    self.labels.append(1)  # gato → 1

        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def mixup_data(x, y): # x --> data, y --> target/lables
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0) # x = tensor([batch_size, chanels, height, width]); x.size(0) = batch size
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam): # compute loss criteria under mixed stuff
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)


def train():

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    
    catdog_dir = Path("/opt/catdog-data")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),      # igualamos tamaños
        transforms.RandomHorizontalFlip(),# voltea imágenes aleatoriamente
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),      # gira hasta ±10 grados
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2),  # cambios leves en luz y contraste
        transforms.ToTensor(),               # pasa a tensor (C x H x W)
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalizar valores entre -1 y 1 
    ])


    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),      # igualamos tamaños
        transforms.ToTensor(),               # pasa a tensor (C x H x W)
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalizar valores entre -1 y 1 
    ])


    train_dataset = DogsCatsDataset(
            data_dir=catdog_dir, split="train", transform=train_transform)

    val_dataset = DogsCatsDataset(
            data_dir=catdog_dir, split="validation", transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
        )

    best_accuracy = 0.0

    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
                train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(
                        criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar(
                'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(
                f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch,
                    'classes': train_dataset.classes
                }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')
            

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


@app.local_entrypoint()
def main():
    train.remote()