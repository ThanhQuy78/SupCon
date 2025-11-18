import math
import torch
from sklearn.neighbors import KNeighborsClassifier

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        # Warmup
        if epoch < self.warmup_epochs:
            factor = epoch / float(self.warmup_epochs)
            return [self.base_lr * factor for _ in self.optimizer.param_groups]

        # Cosine decay
        t = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * t))

        return [self.base_lr * cosine for _ in self.optimizer.param_groups]

def extract_features(model, loader, device):
    encoder = model[0] 
    encoder.eval()

    feats, labs = [], []

    with torch.no_grad():
        for (img1, img2), label in loader:
            img1 = img1.to(device)
            f = encoder(img1)   
            feats.append(f.cpu())
            labs.append(label.cpu())

    return torch.cat(feats), torch.cat(labs)



def knn_accuracy(model, train_loader, test_loader, device, k=5):
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train.numpy(), y_train.numpy())

    acc = knn.score(X_test.numpy(), y_test.numpy()) * 100
    return acc

