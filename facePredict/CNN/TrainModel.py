import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    device = torch.device("cuda" if torch.cuda.is_available() else exit("GPU yok"))

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    class FaceNetLikeModel(nn.Module):
        def __init__(self, embedding_size=256):
            super(FaceNetLikeModel, self).__init__()

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

    model = FaceNetLikeModel().to(device)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_path = r"C:\Users\umutk\OneDrive\Belgeler\Deneme\MyDataset"
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                            pin_memory=True, num_workers=2, persistent_workers=True)

    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    scaler = torch.amp.GradScaler("cuda")

    train_losses = []



    for epoch in range(50):
        total_loss = 0
        print(f"Epoch {epoch + 1} Başlıyor...")

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)

        for img, label in progress_bar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            batch_size = img.size(0)
            indices = torch.arange(batch_size)
            negative_indices = torch.roll(indices, shifts=1)

            anchor = model(img)
            positive = model(img)
            with torch.no_grad():
                negative = model(img[negative_indices])

            with torch.amp.autocast("cuda"):
                loss = criterion(anchor, positive, negative)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.8f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} Tamamlandı Ortalama Loss: {avg_loss:.8f}")

    torch.save(model.state_dict(), "../../GUI/facenet_like_model.pth")
    print("Model kaydoldu")

    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Eğitim Loss Grafiği')
    plt.show()
