import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

from support.LMDB import LMDBDataset

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
BATCH_SIZE = 32

CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]

LMDB_TEST_PATH = "./lmdbs/test.lmdb"
CHECKPOINT_DIR = "./checkpoints"

# ======================
# TRANSFORMS
# ======================
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# LOAD MODEL
# ======================
def load_resnet152(ckpt_path):
    # use no pretrained weights for inference
    model = resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model

# ======================
# ENSEMBLE INFERENCE
# ======================
@torch.no_grad()
def run_resnet152_ensemble(test_loader):
    models = []
    for i in range(5):
        ckpt = f"{CHECKPOINT_DIR}/resnet152_fold_{i}.pth"
        models.append(load_resnet152(ckpt))

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

    return preds.numpy(), probs.numpy(), labels.numpy()

# ======================
# MAIN
# ======================
def main():
    test_ds = LMDBDataset(
        LMDB_TEST_PATH,
        transform=test_tf
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    preds, probs, labels = run_resnet152_ensemble(test_loader)

    # ======================
    # METRICS
    # ======================
    acc = (preds == labels).mean()

    print("\n===== ResNet-152 Ensemble Test Results =====")
    print(f"Test Accuracy: {acc:.6f}\n")

    print("Predicted class counts:", Counter(preds))
    print("True class counts     :", Counter(labels))

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    # ======================
    # SAMPLE PREDICTIONS
    # ======================
    print("\nSample GT / PRED:")
    for i in range(10):
        print(
            f"GT: {CLASS_NAMES[labels[i]]} "
            f"PRED: {CLASS_NAMES[preds[i]]}"
        )

# ======================
# ENTRY
# ======================
if __name__ == "__main__":
    main()
