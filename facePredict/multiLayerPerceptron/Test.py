import torch
from MLP import FaceMLP
from MLPDataLoader import load_data

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\umutk\OneDrive\Belgeler\dataset2"
    _, test_loader, class_names = load_data(data_dir)

    image_size = 160 * 160 * 3
    hidden_size1 = 512
    hidden_size2 = 256
    output_size = len(class_names)

    model = FaceMLP(image_size, hidden_size1, hidden_size2, output_size).to(device)
    model.load_state_dict(torch.load("mlp_face_model.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Doğruluğu: {100 * correct / total:.2f}%")
