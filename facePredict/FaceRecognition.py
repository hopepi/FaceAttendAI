import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import glob
import pickle
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, embedding_size=256):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5a = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, embedding_size)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3a(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.conv3b(x)), negative_slope=0.1)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn4(self.conv4a(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn4(self.conv4b(x)), negative_slope=0.1)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn5(self.conv5a(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn5(self.conv5b(x)), negative_slope=0.1)
        x = F.max_pool2d(x, 2)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.normalize(x, p=2, dim=1)

model = Model().to(device)
model.load_state_dict(torch.load("facenet_like_model.pth", map_location=device))
model.eval()
print("Model Yüklendi!")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

embeddings_file = "face_embeddings.pkl"
stored_embeddings = {}

def load_embeddings():
    global stored_embeddings
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, "rb") as f:
                stored_embeddings = pickle.load(f)
            for key in stored_embeddings:
                stored_embeddings[key] = np.array(stored_embeddings[key])
            print(f"{len(stored_embeddings)} kişi veritabanına yüklendi")
        except Exception as e:
            print(f"Embedding dosyası yüklenirken hata oluştu: {e}")
            stored_embeddings = {}

def add_all_faces_to_db(dataset_path):
    global stored_embeddings

    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"--{len(person_folders)} kişi bulundu--")

    for person in person_folders:
        person_path = os.path.join(dataset_path, person)
        image_files = glob.glob(os.path.join(person_path, "*.jpg")) + glob.glob(os.path.join(person_path, "*.png"))

        print(f"--{person}: {len(image_files)} resim bulundu--")

        for img_path in image_files:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                embedding = model(image).cpu().numpy().flatten()

            if person in stored_embeddings:
                stored_embeddings[person].append(embedding.tolist())
            else:
                stored_embeddings[person] = [embedding.tolist()]

    with open(embeddings_file, "wb") as f:
        pickle.dump(stored_embeddings, f)

    print(f"Tüm yüzler başarıyla veritabanına eklendi ({len(stored_embeddings)} kişi)")


def evaluate_accuracy(test_dataset_path):
    total_tests = 0
    correct_predictions = 0
    wrong_predictions = 0

    person_folders = [f for f in os.listdir(test_dataset_path) if os.path.isdir(os.path.join(test_dataset_path, f))]

    print(f"{len(person_folders)} kişi için test başlatılıyor")

    for person in person_folders:
        person_path = os.path.join(test_dataset_path, person)
        image_files = glob.glob(os.path.join(person_path, "*.jpg")) + glob.glob(os.path.join(person_path, "*.png"))

        for img_path in image_files:
            total_tests += 1
            predicted_name = recognize_face(img_path)

            if predicted_name == person:
                correct_predictions += 1
            else:
                wrong_predictions += 1
                print(f"Yanlış Eşleşme {img_path} için {predicted_name} bulundu ancak gerçek kişi {person}.")

    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    print(f"Toplam Test: {total_tests}")
    print(f"Doğru Tahminler: {correct_predictions}")
    print(f"Yanlış Tahminler: {wrong_predictions}")
    print(f"Model Doğruluk Oranı: %{accuracy:.8f}")

    return accuracy


def recognize_face(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        embedding = model(image).cpu().numpy().flatten()

    scores = []

    print("--Yüz Karşılaştırmaları--")
    for stored_name, stored_emb in stored_embeddings.items():
        stored_emb_array = np.array(stored_emb)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0)

        best_emb = stored_emb_array[np.argmin(np.linalg.norm(stored_emb_array - embedding, axis=1))]

        stored_emb_tensor = torch.tensor(best_emb, dtype=torch.float32, device=device).unsqueeze(0)
        cosine_score = F.cosine_similarity(embedding_tensor, stored_emb_tensor).item()

        l2_score = -torch.norm(embedding_tensor - stored_emb_tensor, p=2).item()

        final_score = (cosine_score + (1 - abs(l2_score))) / 2

        scores.append((stored_name, final_score))

        print(f"---{stored_name} | Cosine: {cosine_score:.8f} | L2: {l2_score:.8f} | Final Skor: {final_score:.8f}---")

    scores.sort(key=lambda x: x[1], reverse=True)
    top_3 = scores[:3]

    avg_score = np.mean([s[1] for s in top_3])
    best_match = top_3[0][0]

    if avg_score > 0.7:
        print(f"En yüksek eşleşme: {best_match} (Skor: {avg_score:.4f})")
        return best_match
    else:
        print("Hiçbir eşleşme bulunamadı döndü.")
        return "Bilinmiyor"


load_embeddings()


test_dataset_path = r"C:\Users\umutk\OneDrive\Belgeler\dataset"
evaluate_accuracy(test_dataset_path)

print("\n🔍 Tanıma Testi:")
test_result = recognize_face("umut.jpg")
print("🎯 Tanıma Sonucu:", test_result)
