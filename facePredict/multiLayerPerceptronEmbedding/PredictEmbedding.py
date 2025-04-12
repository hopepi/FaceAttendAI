import torch
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torchvision.transforms as transforms
from MLPEmbedding import FaceEmbeddingMLP
from MLPDataLoaderEmbedding import load_data

def predict_face(image_path, data_dir=r"C:\Users\umutk\OneDrive\Belgeler\dataset2", embedding_file="train_embeddings.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, class_names = load_data(data_dir)
    image_size = 160 * 160 * 3
    hidden_size1 = 1028
    hidden_size2 = 512
    hidden_size3 = 256
    hidden_size4 = 128
    hidden_size5 = 64
    output_size = len(class_names)

    model = FaceEmbeddingMLP(image_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5,
                             output_size).to(device)
    model.load_state_dict(torch.load("face_embedding_mlp.pth"))
    model.eval()

    with open(embedding_file, 'rb') as f:
        saved_embeddings, saved_labels = pickle.load(f)

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.resnet(image)

    cosine_similarities = cosine_similarity(embedding.cpu().numpy(), saved_embeddings)
    predicted_label_idx = np.argmax(cosine_similarities)

    predicted_name = class_names[saved_labels[predicted_label_idx]]

    print(f"Tanımlanan Kişi: {predicted_name}")
    return predicted_name
