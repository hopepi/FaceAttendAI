import torch
import torch.optim as optim
import torch.nn as nn
from MLP import FaceMLP
from MLPDataLoader import load_data

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\umutk\OneDrive\Belgeler\dataset2"
    train_loader, test_loader, class_names = load_data(data_dir)

    image_size = 160 * 160 * 3
    hidden_size1 = 512
    hidden_size2 = 256
    output_size = len(class_names)

    model = FaceMLP(image_size, hidden_size1, hidden_size2, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
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

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "mlp_face_model.pth")
    print("✅ Model başarıyla kaydedildi.")
