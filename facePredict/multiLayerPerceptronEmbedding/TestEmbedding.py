import torch
from MLPEmbedding import FaceEmbeddingMLP
from MLPDataLoaderEmbedding import load_data

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\umutk\OneDrive\Belgeler\dataset2"
    _, test_loader, class_names = load_data(data_dir)

    image_size = 160 * 160 * 3
    hidden_size1 = 1028
    hidden_size2 = 512
    hidden_size3 = 256
    hidden_size4 = 128
    hidden_size5 = 64
    output_size = len(class_names)

    model = FaceEmbeddingMLP(image_size, hidden_size1, hidden_size2,hidden_size3,hidden_size4,hidden_size5, output_size).to(device)
    model.load_state_dict(torch.load("face_embedding_mlp.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Doğruluğu: {100 * correct / total:.2f}%")
