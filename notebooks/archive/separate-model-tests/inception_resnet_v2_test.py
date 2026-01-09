import torch
import timm
import torch
from torchvision import transforms
from support.LMDB import LMDBDataset

NUM_CLASSES = 3
NUM_FOLDS = 5
BATCH_SIZE = 32
LMDB_ROOT = "./lmdbs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def run_inception_ensemble(test_loader):
    models = []

    for i in range(NUM_FOLDS):
        model = timm.create_model(
            "inception_resnet_v2",
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(
            torch.load(f"checkpoints/inception_resnet_v2_fold_{i}.pth",
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


test_ds = LMDBDataset(f"{LMDB_ROOT}/test.lmdb", transform=train_tf)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

preds, probs, acc = run_inception_ensemble(test_loader)

print(f"Inception-ResNet-V2 Ensemble Accuracy: {acc:.4f}")