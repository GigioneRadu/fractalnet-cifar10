import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from src.architecture import FullFractalNet

def main():
    # Asigurăm existența folderului pentru modelul salvat
    os.makedirs('models', exist_ok=True)

    # 1. Pipeline de date (Data Augmentation pentru rezultate mai bune)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Descărcăm și pregătim dataset-ul CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # BATCH SIZE 256: Aici simți puterea plăcii A4000!
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    # 2. Inițializarea Modelului
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Antrenăm pe dispozitivul: {device}")
    
    model = FullFractalNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Bucla de Antrenare
    epochs = 15 # Durează doar câteva minute pe A4000
    print("Începem antrenarea...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculăm acuratețea pe loc
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        print(f"Epoca [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Acuratețe: {epoch_acc:.2f}%")

    # 4. Salvarea Greutăților
    save_path = 'models/fractalnet_cifar10.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel antrenat și salvat cu succes la: {save_path} 🚀")

if __name__ == '__main__':
    main()