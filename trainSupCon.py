import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.dataset import create_dataloaders
from model.model import ResNetEncoder, ProjectionHead
from loss.loss import SupConLoss
from utils.utils import WarmupCosineLR
from utils.utils import knn_accuracy
from torch.amp import GradScaler, autocast


def train_one_epoch(model,loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss=0.0
    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        optimizer.zero_grad()

        (img1,img2), label = batch
        img1,img2, label = img1.to(device), img2.to(device), label.to(device)
        with autocast(device_type = 'cuda'):  
            images = torch.cat([img1,img2], dim=0)
            features = model(images)
            B = label.shape[0]
            features = features.view(B,2,-1)
            loss = criterion(features, label)
        scaler.scale(loss).backward()      
        scaler.unscale_(optimizer)
        scaler.step(optimizer)           
        scaler.update()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return total_loss/len(loader)



# ================== MAIN ==================
def main():
    data_root = "2011/Fingerprint/Training/MergedDataset"
    NUM_CLASSES = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Parameters
    batch_size = 64
    lr = 0.5 * batch_size / 256
    base_lr = lr
    backbone="resnet101"
    epochs = 50
    save_path= backbone + '_model.pth'

    print("Training Supervised Contrastive Model...")

    train_loader,test_loader = create_dataloaders(data_root,batch_size=batch_size,supcon=True)


    encoder = ResNetEncoder(backbone= backbone, pretrained=True).to(device)
    head = ProjectionHead(encoder.feature_dim, 512, 128).to(device)

    model = nn.Sequential(encoder, head).to(device)
    criterion = SupConLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, max_epochs=epochs, base_lr=base_lr)

    scaler = GradScaler()

    best_train_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,scaler)
        scheduler.step()
            
            
        print(f"Train Loss: {train_loss:.4f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss

            torch.save(encoder.state_dict(), save_path)
            print("Saved encoder to:", save_path)
        if epoch % 5 == 0:
            acc = knn_accuracy(model, train_loader, test_loader, device)
            print(f"KNN Accuracy = {acc:.2f}%")

            

    print("\n SupCon Pretraining Done!")
    exit()

if __name__ == "__main__":
    main()