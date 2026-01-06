import torch
from torch import nn
from support.LMDB import LMDBDataset
from torchvision import transforms
from torchvision.models import densenet201

NUM_CLASSES = 3
LMDB_ROOT = "./lmdbs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def run_densenet201_ensemble(test_loader):
    models = []

    for i in range(5):
        model = densenet201(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES
        )

        model.load_state_dict(
            torch.load(
                f"checkpoints/densenet201_fold_{i}.pth",
                map_location=DEVICE
            )
        )

        model.to(DEVICE)
        model.eval()
        models.append(model)

    all_probs = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        fold_probs = []

        for model in models:
            outputs = model(images)
            fold_probs.append(
                torch.softmax(outputs, dim=1)
            )

        avg_probs = torch.mean(
            torch.stack(fold_probs), dim=0
        )

        all_probs.append(avg_probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    preds = probs.argmax(1)
    acc = (preds == labels).float().mean().item()

    return preds.numpy(), probs.numpy(), labels.numpy(), acc


test_ds = LMDBDataset(
    f"{LMDB_ROOT}/test.lmdb",
    transform=test_tf
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

preds, probs, labels, acc = run_densenet201_ensemble(test_loader)

print(f"DenseNet-201 Ensemble Accuracy: {acc:.4f}")