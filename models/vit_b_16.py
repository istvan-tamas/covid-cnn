from datetime import datetime
import torch
from torch import nn
from torchvision.models import vit_b_16
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from support.LMDB import LMDBDataset

torch.backends.cudnn.benchmark = True

NUM_FOLDS = 5
NUM_CLASSES = 3
EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 32
LMDB_ROOT = "./lmdbs"
LR = 1e-4
WD = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def create_vit():
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        NUM_CLASSES
    )
    return model.to(DEVICE)

scaler = GradScaler("cuda")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_vit_5fold():
    for fold in range(NUM_FOLDS):
        print(f"\n===== ViT FOLD {fold} =====")

        train_ds = LMDBDataset(f"{LMDB_ROOT}/fold_{fold}_train.lmdb", transform=train_tf)
        val_ds = LMDBDataset(f"{LMDB_ROOT}/fold_{fold}_val.lmdb", transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True
        )

        model = create_vit()

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=PATIENCE)

        best_val = 0.0

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, criterion
            )

            scheduler.step(val_acc)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {tr_loss:.4f} | "
                f"Train Acc: {tr_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val:
                best_val = val_acc
                torch.save(
                    model.state_dict(),
                    f"checkpoints/vit_fold_{fold}.pth"
                )

start = datetime.now() #timing!

train_vit_5fold()

print("Training completed in: " + str(datetime.now() - start))