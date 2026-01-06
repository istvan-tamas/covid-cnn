import torch
from torchvision.models import densenet121
from torchvision import transforms
from torch.utils.data import DataLoader
from support.LMDB import LMDBDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

torch.backends.cudnn.benchmark = True

NUM_CLASSES = 3
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
    f"./checkpoints/densenet_fold_{i}.pth" for i in range(5)
]

preds, probs, test_acc = run_test_ensemble(test_loader, model_paths)

print(f"Ensemble Test Accuracy: {test_acc:.4f}")
