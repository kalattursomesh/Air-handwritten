import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

# ============================================================
# CNN MODEL
# ============================================================
class HandwritingCNN(nn.Module):
    """Lightweight CNN for single character recognition."""
    
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================
# DATASET
# ============================================================
class DrawingDataset(Dataset):
    """Dataset of user-drawn characters."""
    
    def __init__(self, data_dir="training_data"):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.data_dir = data_dir
        
        if os.path.exists(data_dir):
            self._load_data()
    
    def _load_data(self):
        """Load all saved drawings."""
        map_file = os.path.join(self.data_dir, "label_map.json")
        if os.path.exists(map_file):
            with open(map_file, "r") as f:
                self.label_map = json.load(f)
        
        for label_name, label_idx in self.label_map.items():
            label_dir = os.path.join(self.data_dir, label_name)
            if os.path.exists(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.endswith(".npy"):
                        filepath = os.path.join(label_dir, fname)
                        img = np.load(filepath)
                        self.samples.append(img)
                        self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples[idx].astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # Add channel dim
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ============================================================
# TRAINER
# ============================================================
class ModelTrainer:
    """Handles training and prediction."""
    
    def __init__(self, data_dir="training_data", model_path="handwriting_model.pth"):
        self.data_dir = data_dir
        self.model_path = model_path
        self.model = None
        self.label_map = {}
        self.reverse_map = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing label map
        map_file = os.path.join(data_dir, "label_map.json")
        if os.path.exists(map_file):
            with open(map_file, "r") as f:
                self.label_map = json.load(f)
                self.reverse_map = {v: k for k, v in self.label_map.items()}
        
        # Load existing model
        if os.path.exists(model_path) and self.label_map:
            self.model = HandwritingCNN(num_classes=len(self.label_map))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"[MODEL] Loaded with {len(self.label_map)} classes.")
    
    def save_sample(self, image_array, label):
        """Save a 28x28 grayscale drawing with its label."""
        if label not in self.label_map:
            self.label_map[label] = len(self.label_map)
            self.reverse_map[self.label_map[label]] = label
        
        # Save label map
        map_file = os.path.join(self.data_dir, "label_map.json")
        with open(map_file, "w") as f:
            json.dump(self.label_map, f)
        
        # Save image
        label_dir = os.path.join(self.data_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
        filepath = os.path.join(label_dir, f"sample_{count}.npy")
        np.save(filepath, image_array)
        
        return count + 1
    
    def get_stats(self):
        """Return training data statistics."""
        stats = {}
        for label in self.label_map:
            label_dir = os.path.join(self.data_dir, label)
            if os.path.exists(label_dir):
                count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
                stats[label] = count
        return stats
    
    def train(self, epochs=30):
        """Train the CNN on collected data."""
        dataset = DrawingDataset(self.data_dir)
        
        if len(dataset) < 5:
            return {"success": False, "message": "Need at least 5 samples to train."}
        
        num_classes = len(self.label_map)
        self.model = HandwritingCNN(num_classes=num_classes).to(self.device)
        
        # Data augmentation through multiple passes
        loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            scheduler.step()
            accuracy = 100 * correct / total
            losses.append(total_loss)
        
        # Save model
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        
        return {
            "success": True,
            "message": f"Trained on {len(dataset)} samples, {num_classes} classes",
            "accuracy": round(accuracy, 1),
            "epochs": epochs
        }
    
    def predict(self, image_array):
        """Predict a character from a 28x28 grayscale image."""
        if self.model is None:
            return {"success": False, "prediction": "?", "confidence": 0, 
                    "message": "No trained model. Train first!"}
        
        img = image_array.astype(np.float32) / 255.0
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_idx = predicted.item()
            conf = confidence.item() * 100
            
            label = self.reverse_map.get(pred_idx, "?")
            
            # Top 3 predictions
            top3_conf, top3_idx = torch.topk(probabilities, min(3, probabilities.size(1)))
            top3 = []
            for i in range(top3_conf.size(1)):
                idx = top3_idx[0][i].item()
                c = top3_conf[0][i].item() * 100
                lbl = self.reverse_map.get(idx, "?")
                top3.append({"label": lbl, "confidence": round(c, 1)})
        
        return {
            "success": True,
            "prediction": label,
            "confidence": round(conf, 1),
            "top3": top3
        }
