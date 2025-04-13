import torch
import torch.optim as optim
import torch.nn as nn
from MLP import FaceMLP
from MLPDataLoader import load_data
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\umutk\OneDrive\Belgeler\dataset2"
    train_loader, test_loader, class_names = load_data(data_dir)

    image_size = 160 * 160 * 3
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = len(class_names)

    model = FaceMLP(image_size, hidden_size1, hidden_size2, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 100
    best_loss = float('inf')
    patience = 10
    counter = 0

    loss_values = []  # loss değerlerini toplamak için

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)  # Her epoch sonunda loss kaydedildi

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Yeni en iyi model kaydedildi.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping tetiklendi.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_values)+1), loss_values, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Başına Kayıp (Loss) Değeri')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
