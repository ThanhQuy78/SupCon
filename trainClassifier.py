import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.dataset import create_dataloaders
from model.model import ResNetEncoder, ProjectionHead, ClassifierHead


def train_one_epoch(model,encoder, loader, criterion, optimizer, device):
    encoder.eval()
    model.train()
    total_loss=0.0
    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        optimizer.zero_grad()

        images, label = batch
        images, label = images.to(device), label.to(device)

        with torch.no_grad():
            features = encoder(images)
        logits = model(features)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return total_loss/len(loader)

def validate(model,encoder, loader, criterion, device):
    encoder.eval()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

# ================== MAIN ==================
def main():
    data_root = "2011/Fingerprint/Training/MergedDataset"
    NUM_CLASSES = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Parameters
    batch_size = 64
    lr = 1e-3
    backbone="resnet101"
    epochs = 20
    save_path= 'classifier.pth'
    
    print("Training Classifier Head...")

    train_loader,test_loader = create_dataloaders(data_root,batch_size=batch_size,supcon=False)


    encoder = ResNetEncoder(backbone= backbone, pretrained=True).to(device)
    encoder.load_state_dict(torch.load(backbone + '_model.pth', map_location=device))

    for param in encoder.parameters():
        param.requires_grad = False

    classifier_head = ClassifierHead(encoder.feature_dim, NUM_CLASSES).to(device)
    model = classifier_head.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        print(f"Epoch [{epoch}/{epochs}]")
        train_loss = train_one_epoch(model,encoder, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model,encoder, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), save_path)
            print("Saved classifier head to:", save_path)

if __name__ == "__main__":
    main()