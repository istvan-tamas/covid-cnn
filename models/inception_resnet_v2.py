from datetime import datetime
import torch
import timm
import torch
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

train_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def create_model():
    model = timm.create_model(
        "inception_resnet_v2",
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    return model.to(DEVICE)

def train_one_fold(fold_idx, train_ds, val_ds):
    model = create_model()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=PATIENCE)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # ===== TRAIN =====
        model.train()
        correct, total, loss_sum = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ===== VALIDATE =====
        model.eval()
        correct, total, val_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)

        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                f"checkpoints/inception_resnet_v2_fold_{fold_idx}.pth"
            )

    return best_acc

start = datetime.now() #timing!

for fold in range(NUM_FOLDS):
    print(f"\n===== Fold {fold} =====")

    train_ds = LMDBDataset(
        f"{LMDB_ROOT}/fold_{fold}_train.lmdb", transform=train_transform
    )
    val_ds = LMDBDataset(
        f"{LMDB_ROOT}/fold_{fold}_val.lmdb", transform=val_transform
    )

    train_one_fold(fold, train_ds, val_ds)
    
print("Training completed in: " + str(datetime.now() - start))