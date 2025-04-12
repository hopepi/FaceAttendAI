import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from MLPDataLoaderEmbedding import load_data
from MLPEmbedding import FaceEmbeddingMLP
import os
import pickle
import numpy as np

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\umutk\OneDrive\Belgeler\dataset2"
    train_loader, test_loader, class_names = load_data(data_dir)

    image_size = 160 * 160 * 3
    hidden_size1 = 1028
    hidden_size2 = 512
    hidden_size3 = 256
    hidden_size4 = 128
    hidden_size5 = 64
    output_size = len(class_names)

    model = FaceEmbeddingMLP(image_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5,
                             output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    save_embeddings(train_loader, model, output_file="train_embeddings.pkl")

    num_epochs = 125
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "face_embedding_mlp.pth")
    print("Model başarıyla kaydedildi.")




def save_embeddings(data_loader, model, output_file="embeddings.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device tanımlaması
    embeddings = []
    labels = []

    # Modeli eval moduna alıyoruz
    model.eval()

    with torch.no_grad():
        for images, label in data_loader:
            images = images.to(device)  # Modeli cihazda çalıştırmak için

            # Embedding çıkartma
            embedding = model.resnet(images)  # Özellik çıkarma
            embeddings.append(embedding.cpu().numpy())  # CPU'ya taşıyoruz
            labels.append(label.cpu().numpy())  # Etiketleri de CPU'ya taşıyoruz

    # Embedding'leri ve etiketleri kaydediyoruz
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Kaydetme işlemi
    with open(output_file, 'wb') as f:
        pickle.dump((embeddings, labels), f)

    print(f"Embeddingler {output_file} dosyasına kaydedildi.")