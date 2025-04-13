import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from MLPDataLoaderEmbedding import load_data
from MLPEmbedding import FaceEmbeddingMLP
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    loss_values = []

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

        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "face_embedding_mlp.pth")
    print("Model başarıyla kaydedildi.")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_values)+1), loss_values, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Başına Kayıp (Loss) Değeri')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_embeddings(data_loader, model, output_file="embeddings.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, label in data_loader:
            images = images.to(device)

            embedding = model.resnet(images)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    with open(output_file, 'wb') as f:
        pickle.dump((embeddings, labels), f)

    print(f"Embeddingler {output_file} dosyasına kaydedildi.")
