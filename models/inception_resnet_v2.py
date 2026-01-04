import torch
import timm
import torch
from torchvision import transforms
from support.LMDB import LMDBDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
EPOCHS = 15
BATCH_SIZE = 32

from torchvision import transforms

train_tf = transforms.Compose([
    transforms.Resize(320),
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
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


from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2)
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
                f"checkpoints/inception_resnet_fold_{fold_idx}.pt"
            )

    return best_acc


for fold in range(5):
    print(f"\n===== Fold {fold} =====")

    train_ds = LMDBDataset(
        f"./lmdbs/fold_{fold}_train.lmdb", transform=train_tf
    )
    val_ds = LMDBDataset(
        f"./lmdbs/fold_{fold}_val.lmdb", transform=val_tf
    )

    train_one_fold(fold, train_ds, val_ds)



@torch.no_grad()
def run_inception_ensemble(test_loader):
    models = []

    for i in range(5):
        model = timm.create_model(
            "inception_resnet_v2",
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(
            torch.load(f"checkpoints/inception_resnet_fold_{i}.pt",
                       map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()
        models.append(model)

    all_probs = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        probs = []
        for model in models:
            out = model(images)
            probs.append(torch.softmax(out, dim=1))

        avg_probs = torch.mean(torch.stack(probs), dim=0)

        all_probs.append(avg_probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    preds = probs.argmax(1)
    acc = (preds == labels).float().mean().item()

    return preds.numpy(), probs.numpy(), acc


test_ds = LMDBDataset("./lmdbs/test.lmdb", transform=val_tf)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=32, shuffle=False,
    num_workers=4, pin_memory=True
)

preds, probs, acc = run_inception_ensemble(test_loader)

print(f"Inception-ResNet-V2 Ensemble Accuracy: {acc:.4f}")