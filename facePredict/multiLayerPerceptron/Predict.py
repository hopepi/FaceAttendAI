import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from MLP import FaceMLP
from MLPDataLoader import load_data

def predict_face(image_path, data_dir=r"C:\Users\umutk\OneDrive\Belgeler\dataset2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, class_names = load_data(data_dir)
    output_size = len(class_names)

    image_size = 160 * 160 * 3
    hidden_size1 = 128
    hidden_size2 = 64

    model = FaceMLP(image_size, hidden_size1, hidden_size2, output_size).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image.view(image.size(0), -1))
        probabilities = F.softmax(output, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    threshold = 0.2
    if max_prob.item() < threshold:
        print("Bu yüz veritabanında yok (tanınmadı).")
        return "Bilinmeyen"

    predicted_id = predicted.item()
    predicted_name = class_names[predicted_id]
    print(f"Tanımlanan Kişi: {predicted_name} (ID: {predicted_id})")
    return predicted_name
