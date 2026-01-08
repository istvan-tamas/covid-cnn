import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    alexnet,
    densenet121, densenet201,
    resnet152
)
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import numpy as np

# Optional: use timm for InceptionResNetV2
import timm
from support.LMDB import LMDBDataset

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_FOLDS = 5

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
# LMDBDataset placeholder
# ======================
# You already have LMDBDataset defined elsewhere
# from your_dataset_module import LMDBDataset

# ======================
# MODEL LOADING FUNCTIONS
# ======================
def load_alexnet(ckpt_path):
    model = alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_densenet121(ckpt_path):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_densenet201(ckpt_path):
    model = densenet201(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_resnet152(ckpt_path):
    model = resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_inceptionresnetv2(ckpt_path):
    # Using timm implementation
    model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ======================
# ENSEMBLE FUNCTION
# ======================
@torch.no_grad()
def run_ensemble(test_loader, load_model_fn, model_name):
    models = [load_model_fn(f"{CHECKPOINT_DIR}/{model_name}_fold_{i}.pth") for i in range(NUM_FOLDS)]

    all_probs = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        fold_probs = []
        for model in models:
            outputs = model(images)
            fold_probs.append(torch.softmax(outputs, dim=1))

        avg_probs = torch.mean(torch.stack(fold_probs), dim=0)
        all_probs.append(avg_probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = probs.argmax(dim=1)

    return preds.numpy(), probs.numpy(), labels.numpy()

# ======================
# RUN TEST
# ======================
def main():
    test_ds = LMDBDataset(LMDB_TEST_PATH, transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model_fns = {
        "alexnet": load_alexnet,
        "densenet_121": load_densenet121,
        "densenet201": load_densenet201,
        "resnet152": load_resnet152,
        "inception_resnet_v2": load_inceptionresnetv2
    }

    for model_name, fn in model_fns.items():
        print(f"\n===== {model_name.upper()} 5-FOLD ENSEMBLE TEST =====")
        preds, probs, labels = run_ensemble(test_loader, fn, model_name)

        acc = (preds == labels).mean()
        print(f"Test Accuracy: {acc:.6f}")

        print("Predicted class counts:", Counter(preds))
        print("True class counts     :", Counter(labels))

        cm = confusion_matrix(labels, preds)
        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(
            labels, preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        ))

        print("\nSample GT / PRED:")
        for i in range(min(10, len(labels))):
            print(f"GT: {CLASS_NAMES[labels[i]]} PRED: {CLASS_NAMES[preds[i]]}")

# ======================
# ENTRY
# ======================
if __name__ == "__main__":
    main()
