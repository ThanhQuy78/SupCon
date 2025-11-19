import torch
import torch.nn as nn
from model.model import ResNetEncoder, ProjectionHead, ClassifierHead
from dataset.dataset import FingerPrintDataset, get_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = "resnet101"

NUM_CLASSES = 8
BATCH_SIZE = 64
IMAGE_SIZE = 224

encoder_path = backbone + "_model.pth"
classifier_path = "classifier_best.pth"
data_root = "2011/Fingerprint/Testing/ItaldataTest/ItaldataTest/Spoof"


def main():
    transform = get_transforms(image_size=IMAGE_SIZE, supcon=False)
    test_dataset = FingerPrintDataset(root_dir=data_root, transform=transform, two_view=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    encoder = ResNetEncoder(backbone=backbone, pretrained=False).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location= device), strict=False)
    encoder.eval()

    classifier_head = ClassifierHead(encoder.feature_dim, NUM_CLASSES).to(device)
    classifier_head.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier_head.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            logits = classifier_head(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            #labels = (labels != 0).long()              #Bỏ comment để chuyển sang bài toán binary classification
            #predicted = (predicted != 0).long()        
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = total_loss / total

    print(f"Test Accuracy (multiclass classifier): {acc:.2f}%")

if __name__ == "__main__":
    main()