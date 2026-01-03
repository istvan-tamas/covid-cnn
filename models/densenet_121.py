import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
import numpy as np
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader
from support.LMDB import LMDBDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


def build_densenet(num_classes=3, freeze_backbone=True):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    model.classifier = nn.Linear(1024, num_classes)
    return model

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def make_weighted_loss(dataset, device):
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    total = sum(counts.values())

    weights = torch.tensor(
        [total / counts[i] for i in range(len(counts))],
        dtype=torch.float,
        device=device
    )

    return nn.CrossEntropyLoss(weight=weights)



def make_optimizer(model, lr):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

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

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

NUM_FOLDS = 5
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 64
LMDB_ROOT = "./lmdbs"
NUM_CLASSES = 3

fold_best_acc = []

for fold in range(NUM_FOLDS):
    print(f"\n===== Fold {fold} =====")

    train_ds = LMDBDataset(
        f"{LMDB_ROOT}/fold_{fold}_train.lmdb",
        transform=train_transform
    )
    val_ds = LMDBDataset(
        f"{LMDB_ROOT}/fold_{fold}_val.lmdb",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = build_densenet(num_classes=3, freeze_backbone=True).to(DEVICE)

    criterion = make_weighted_loss(train_ds, DEVICE)
    optimizer = make_optimizer(model, lr=1e-4)
    scheduler = make_scheduler(optimizer)

    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                model.state_dict(),
                f"densenet_fold_{fold}.pth"
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping")
                break

    fold_best_acc.append(best_val_acc)
    


print("\n===== 5-Fold CV Results =====")
for i, acc in enumerate(fold_best_acc):
    print(f"Fold {i}: {acc:.4f}")

print(f"Mean: {np.mean(fold_best_acc):.4f}")
print(f"Std:  {np.std(fold_best_acc):.4f}")

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_densenet(checkpoint_path):
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(
        model.classifier.in_features, NUM_CLASSES
    )

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def run_test_ensemble(test_loader, model_paths):
    models = [load_densenet(p) for p in model_paths]

    all_probs = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        fold_probs = []

        for model in models:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            fold_probs.append(probs)

        avg_probs = torch.mean(torch.stack(fold_probs), dim=0)

        all_probs.append(avg_probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    preds = probs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    return preds.numpy(), probs.numpy(), acc




test_ds = LMDBDataset("./lmdbs/test.lmdb", transform=test_tf)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model_paths = [
    f"densenet_fold_{i}.pth" for i in range(5)
]

preds, probs, test_acc = run_test_ensemble(test_loader, model_paths)

print(f"Ensemble Test Accuracy: {test_acc:.4f}")
