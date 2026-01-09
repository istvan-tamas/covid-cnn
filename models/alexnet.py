import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from support.LMDB import LMDBDataset

torch.backends.cudnn.benchmark = True

NUM_FOLDS = 5
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LMDB_ROOT = "./lmdbs"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# running the model on the folds
fold_results = []

for fold in range(NUM_FOLDS):
    print(f"\n===== Fold {fold} =====")

    train_lmdb = f"{LMDB_ROOT}/fold_{fold}_train.lmdb"
    val_lmdb = f"{LMDB_ROOT}/fold_{fold}_val.lmdb"

    train_dataset = LMDBDataset(train_lmdb, transform=train_transform)
    val_dataset = LMDBDataset(val_lmdb, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Model
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)

    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"checkpoints/alexnet_fold_{fold}.pth"
            )

    fold_results.append(best_val_acc)