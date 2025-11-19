import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import ResNetEncoder
from dataset.dataset import FingerPrintDataset, get_transforms
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = "resnet50"
NUM_CLASSES = 8
IMAGE_SIZE = 224

encoder_path = backbone +"_model.pth"
data_root = "2011/Fingerprint/Testing/DigitalTest/DigitalTest/Spoof"

samples_per_class = 5

transform = get_transforms(image_size=IMAGE_SIZE, supcon=False)
full_dataset = FingerPrintDataset(root_dir=data_root, transform=transform, two_view=False)

encoder = ResNetEncoder(backbone=backbone, pretrained=False).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=False)
encoder.eval()

present_classes = set()
for _, label in full_dataset:
    present_classes.add(label)


class_features = {i: [] for i in present_classes}
class_count = {i: 0 for i in present_classes}


indices = list(range(len(full_dataset)))
random.shuffle(indices)

with torch.no_grad():
    for idx in indices:
        img, label = full_dataset[idx]
        if class_count[label] >= samples_per_class:
            continue
        img = img.unsqueeze(0).to(device)
        feat = encoder(img)[0].cpu()
        class_features[label].append(feat)
        class_count[label] += 1
        if all([class_count[c] >= samples_per_class for c in present_classes]):
            break

test_dataset = FingerPrintDataset(root_dir=data_root, transform=transform, two_view=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        f_test = encoder(img)[0].cpu()
        sims = []
        class_list = sorted(present_classes)

        for c in class_list:
            refs = torch.stack(class_features[c])
            f_expand = f_test.unsqueeze(0).expand(refs.size(0), -1)
            cosine = F.cosine_similarity(f_expand, refs, dim=1)
            sims.append(cosine.mean().item())

        pred = class_list[int(np.argmax(sims))]
        labels = label.item()
        #pred = 1 if pred != 0 else 0           #Bỏ comment để chuyển sang bài toán binary classification
        #labels = 1 if labels != 0 else 0
        if pred == labels:
            correct += 1
        total += 1

acc = 100 * correct / total
print(f"Test Accuracy (cosine similarity) : {acc:.2f}%")
