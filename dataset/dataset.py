import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ================== DATASET ==================
class FingerPrintDataset(Dataset):
    def __init__(self, root_dir, transform=None, two_view=False):
        self.root_dir = root_dir
        self.transform = transform
        self.two_view = two_view

        self.class_to_idx = {
            "Live": 0, "EcoFlex": 1, "Gelatin": 2, "Latex": 3,
            "Playdoh": 4, "Silgum": 5, "WoodGlue": 6, "Silicone": 7
        }

        self.samples = []
        for dirpath, _, filenames in os.walk(root_dir):
            class_name = os.path.basename(dirpath)
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            for fname in filenames:
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                    self.samples.append((os.path.join(dirpath,fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path,label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.two_view and self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
            return (img1,img2), label
        elif self.transform:
            img = self.transform(img)
        return img, label


# ================== TRANSFORMS ==================
def get_transforms(image_size=224, supcon=False):
    if supcon:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])


def create_dataloaders(root_dir, batch_size=32, image_size=224, supcon=False, test_split=0.2):
    transform = get_transforms(image_size, supcon)
    dataset = FingerPrintDataset(root_dir, transform=transform, two_view=supcon)
    total_len = len(dataset)
    train_len = int(total_len*(1-test_split))
    test_len = total_len - train_len
    train_ds, test_ds = random_split(dataset, [train_len,test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


